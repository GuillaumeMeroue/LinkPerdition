import torch
import numpy as np
from kge.utils import set_seed
from kge.data import get_dataloader, PrecomputedNegativeSampler, ReproductibleOnTheFlyNegativeSampler
from tqdm import tqdm
import wandb
import time
import os


def get_loss_fn(loss_type, margin=None):
    if loss_type == 'bce':
        return torch.nn.BCEWithLogitsLoss()
    elif loss_type == 'margin':
        assert margin is not None, 'Margin must be specified for margin loss.'
        return torch.nn.MarginRankingLoss(margin=margin)
    elif loss_type == 'ce':
        return torch.nn.CrossEntropyLoss()
    else:
        # Raise NotImplementedError for unsupported loss types
        raise NotImplementedError(f"Loss type '{loss_type}' is not implemented. Supported: 'bce', 'margin', 'ce'.")


def apply_regularization(model, reg_type, reg_entity_weight=0.0, reg_relation_weight=0.0):
    reg = 0.0
    if reg_type == 'none':
        return 0.0
    if hasattr(model, 'entity_emb'):
        ent = model.entity_emb.weight
        if reg_type == 'l1':
            reg += reg_entity_weight * ent.abs().sum()
        elif reg_type == 'l2':
            reg += reg_entity_weight * (ent ** 2).sum()
        elif reg_type == 'l3':
            reg += reg_entity_weight * (ent.abs() ** 3).sum()
    if hasattr(model, 'relation_emb'):
        rel = model.relation_emb.weight
        if reg_type == 'l1':
            reg += reg_relation_weight * rel.abs().sum()
        elif reg_type == 'l2':
            reg += reg_relation_weight * (rel ** 2).sum()
        elif reg_type == 'l3':
            reg += reg_relation_weight * (rel.abs() ** 3).sum()
    return reg

def init_training(
    model,
    train_triples,
    valid_triples,
    entity2id,
    relation2id,
    seed_neg,
    seed_order,
    seed_init,
    num_neg_h,
    num_neg_t,
    batch_size,
    lr,
    sampler_type="reproducible_on_the_fly",
    ):
    # set_seed(seed_init)
    # model.reset_parameters() Now, it's done in the constructor
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    # The new get_dataloader expects a numpy array and doesn't need mappings
    train_loader, train_generator = get_dataloader(train_triples, batch_size, seed_order, shuffle=True, return_indices=True)
    valid_loader, valid_generator = get_dataloader(valid_triples, batch_size, seed_order, shuffle=False, return_indices=True)
    if sampler_type == "precomputed":
        neg_sampler = PrecomputedNegativeSampler(
            num_entities=len(entity2id),
            seed=seed_neg,
            num_neg_h=num_neg_h,
            num_neg_t=num_neg_t,
            triples=train_triples
        )
        valid_neg_sampler = PrecomputedNegativeSampler(
            num_entities=len(entity2id),
            seed=42, # It's 42 such that every seed config will have the same validation set
            num_neg_h=num_neg_h,
            num_neg_t=num_neg_t,
            triples=valid_triples
        )
    elif sampler_type == "reproducible_on_the_fly":
        # The sampler now works with numpy arrays
        neg_sampler = ReproductibleOnTheFlyNegativeSampler(
            triples=train_triples,
            num_entities=len(entity2id),
            seed=seed_neg,
            num_neg_h=num_neg_h,
            num_neg_t=num_neg_t,
        )
        valid_neg_sampler = ReproductibleOnTheFlyNegativeSampler(
            triples=valid_triples,
            num_entities=len(entity2id),
            seed=42, # It's 42 such that every seed config will have the same validation set
            num_neg_h=num_neg_h,
            num_neg_t=num_neg_t,
        )
    return model, optimizer, train_loader, valid_loader, neg_sampler, valid_neg_sampler, train_generator, valid_generator

def one_epoch(model, optimizer, data_loader, neg_sampler, epoch, loss_fn, loss_type='bce', margin=None, reg_type='none', reg_entity_weight=0.0, reg_relation_weight=0.0, is_train=True, debug=False):
    """
    Train the model for one epoch using negative sampling and appropriate scoring modes.
    """
    model.train()
    batch_losses = []
    device = next(model.parameters()).device

    grad_mode = torch.set_grad_enabled(is_train)

    batch_orders = []
    all_neg_triples = set()

    for batch in tqdm(data_loader, desc=f"Epoch {epoch}"):
        # The batch is now a tuple of numpy arrays, convert to tensors and move to device
        if len(batch) == 4:
            h_np, r_np, t_np, idx = batch
            # h = torch.from_numpy(h_np).to(device)
            # r = torch.from_numpy(r_np).to(device)
            # t = torch.from_numpy(t_np).to(device)
            h = h_np
            r = r_np
            t = t_np
        else:
            h_np, r_np, t_np = batch
            idx = None
            # h = torch.from_numpy(h_np).to(device)
            # r = torch.from_numpy(r_np).to(device)
            # t = torch.from_numpy(t_np).to(device)
            h = h_np
            r = r_np
            t = t_np

        if debug:
            batch_orders.append(list(zip(h.tolist(), r.tolist(), t.tolist())))

        # The negative sampler now takes numpy arrays and returns numpy arrays
        # The batch passed to the sampler must be on CPU (numpy)
        batch_np = (h_np, r_np, t_np, idx)
        h = h.to(device)
        r = r.to(device)
        t = t.to(device)

        if loss_type == "margin":
            # Tail corruption
            neg_tail_np = neg_sampler.sample(batch_np, mode="tail")
            neg_tail = neg_tail_np.to(device)
            if debug:
                for i in range(neg_tail.size(0)):
                    for neg_t in neg_tail[i, 1:].tolist():
                        all_neg_triples.add((h[i].item(), r[i].item(), neg_t))
            heads, rels, tails = h, r, neg_tail
            scores_tail = model(heads, rels, tails, score_mode="multi_tails")
            pos_scores, neg_scores = scores_tail[:, 0], scores_tail[:, 1:].reshape(-1)
            y = torch.ones_like(neg_scores, device=device)
            loss_tail = loss_fn(pos_scores.repeat_interleave(scores_tail.shape[1]-1), neg_scores, y)

            # Head corruption
            neg_head_np = neg_sampler.sample(batch_np, mode="head")
            neg_head = neg_head_np.to(device)
            if debug:
                for i in range(neg_head.size(0)):
                    for neg_h in neg_head[i, 1:].tolist():
                        all_neg_triples.add((neg_h, r[i].item(), t[i].item()))
            heads_h, rels_h, tails_h = neg_head, r, t
            scores_head = model(heads_h, rels_h, tails_h, score_mode="multi_heads")
            pos_scores_h, neg_scores_h = scores_head[:, 0], scores_head[:, 1:].reshape(-1)
            y_h = torch.ones_like(neg_scores_h, device=device)
            loss_head = loss_fn(pos_scores_h.repeat_interleave(scores_head.shape[1]-1), neg_scores_h, y_h)

            loss = (loss_tail + loss_head) / 2.0
            
        elif loss_type == "bce":
            # Tail corruption
            neg_tail_np = neg_sampler.sample(batch_np, mode="tail")
            neg_tail = neg_tail_np.to(device)
            if debug:
                for i in range(neg_tail.size(0)):
                    for neg_t in neg_tail[i, 1:].tolist():
                        all_neg_triples.add((h[i].item(), r[i].item(), neg_t))
            heads, rels, tails = h, r, neg_tail
            scores_tail = model(heads, rels, tails, score_mode="multi_tails")
            labels_tail = torch.zeros_like(scores_tail, device=device)
            labels_tail[:, 0] = 1.0
            loss_tail = loss_fn(scores_tail, labels_tail)

            # Head corruption
            neg_head_np = neg_sampler.sample(batch_np, mode="head")
            neg_head = neg_head_np.to(device)
            if debug:
                for i in range(neg_head.size(0)):
                    for neg_h in neg_head[i, 1:].tolist():
                        all_neg_triples.add((neg_h, r[i].item(), t[i].item()))
            heads_h, rels_h, tails_h = neg_head, r, t
            scores_head = model(heads_h, rels_h, tails_h, score_mode="multi_heads")
            labels_head = torch.zeros_like(scores_head, device=device)
            labels_head[:, 0] = 1.0
            loss_head = loss_fn(scores_head, labels_head)
            
            loss = (loss_tail + loss_head) / 2.0
        else: # ce
            # --- Tail corruption ---
            neg_tail_np = neg_sampler.sample(batch_np, mode="tail")
            neg_tail = neg_tail_np.to(device)
            if debug:
                for i in range(neg_tail.size(0)):
                    for neg_t in neg_tail[i, 1:].tolist():
                        all_neg_triples.add((h[i].item(), r[i].item(), neg_t))
            num_pos = neg_tail.shape[0]
            heads, rels, tails = h, r, neg_tail
            scores_tail = model(heads, rels, tails, score_mode="multi_tails")
            targets_tail = torch.zeros(num_pos, dtype=torch.long, device=device)
            loss_tail = loss_fn(scores_tail, targets_tail)

            # --- Head corruption ---
            neg_head_np = neg_sampler.sample(batch_np, mode="head")
            neg_head = neg_head_np.to(device)
            if debug:
                for i in range(neg_head.size(0)):
                    for neg_h in neg_head[i, 1:].tolist():
                        all_neg_triples.add((neg_h, r[i].item(), t[i].item()))
            heads_h, rels_h, tails_h = neg_head, r, t
            scores_head = model(heads_h, rels_h, tails_h, score_mode="multi_heads")
            targets_head = torch.zeros(num_pos, dtype=torch.long, device=device)
            loss_head = loss_fn(scores_head, targets_head)
            loss = (loss_tail + loss_head) / 2.0

        reg = apply_regularization(model, reg_type, reg_entity_weight, reg_relation_weight)
        loss = loss + reg
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        batch_losses.append(loss.item())
        
    # return params, batch_orders, all_negs, loss_moyenne_epoch
    loss_moyenne_epoch = float(sum(batch_losses) / len(batch_losses)) if batch_losses else 0.0
    # batch_orders and all_neg_triples are used for testing
    return model.state_dict(), batch_orders, all_neg_triples, loss_moyenne_epoch



def train_epoch_loop(
    model, optimizer, train_loader, valid_loader, neg_sampler, valid_neg_sampler, num_epochs, valid_triples, entity2id, relation2id, eval_every, early_stop_metric, early_stop_patience, early_stop_delta, evaluator,
    train_generator, valid_generator,
    loss_fn, loss_type='bce', margin=None, reg_type='none', reg_entity_weight=0.0, reg_relation_weight=0.0,
    log_to_wandb=True, wandb_run=None, max_hours=float('inf'),
    resume_checkpoint=False, checkpoint_dir='.', checkpoint_every=1
):
    """
    Loop over epochs, calling train_one_epoch and optionally evaluating.
    Saves the best model weights based on validation metrics and restores them when early stopping is triggered.
    """
    best_metric = -float('inf')
    best_epoch = 0
    best_model_state = None
    history = []
    start_epoch = 1

    if resume_checkpoint:
        """
        See tests/test_checkpointing.py were we attest that all the states, and in particular the random states are well restored
        pytest tests/test_checkpointing.py::test_checkpoint_reproducibility -v -s
        """
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
        if os.path.exists(checkpoint_path):
            try:
                print(f"Resuming from checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_metric = checkpoint['best_metric']
                best_epoch = checkpoint['best_epoch']
                best_model_state = checkpoint['best_model_state']
                history = checkpoint['history']
                # Restore forward count for reproducibility
                if 'model_forward_count' in checkpoint:
                    model.forward_count = checkpoint['model_forward_count']
                torch.set_rng_state(checkpoint['torch_rng_state'])
                np.random.set_state(checkpoint['numpy_rng_state'])
                train_generator.set_state(checkpoint['train_loader_generator_state'])
                neg_sampler.set_state(checkpoint['neg_sampler_state'])
                
                # Actually I think it's useless beacause, at the beginning of the model forward,
                # we call set_seed(self.seed_forward + self.forward_count), and both forward_count and seed_forward have been restored. 
                if 'cuda_rng_state' in checkpoint and torch.cuda.is_available():
                    torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
                if 'cuda_rng_state_all' in checkpoint and torch.cuda.is_available() and torch.cuda.device_count() > 1:
                    torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state_all'])
                print(f"Successfully resumed from checkpoint at epoch {checkpoint['epoch']}")
                
            except (RuntimeError, EOFError) as e:
                error_msg = str(e)
                is_corruption = (
                    "PytorchStreamReader failed" in error_msg or 
                    "failed finding central directory" in error_msg or
                    "UnpicklingError" in str(type(e)) or
                    isinstance(e, EOFError)
                )
                if is_corruption:
                    print(f"Corrupted checkpoint detected: {type(e).__name__}: {e}")
                    print(f"Deleting corrupted checkpoint: {checkpoint_path}")
                    try:
                        os.remove(checkpoint_path)
                        print("Corrupted checkpoint deleted successfully.")
                    except Exception as delete_error:
                        print(f"Could not delete corrupted checkpoint: {delete_error}")
                    print("Starting training from scratch...")
                    # Reset to default values
                    start_epoch = 1
                    best_metric = -float('inf')
                    best_epoch = 0
                    best_model_state = None
                    history = []
                else:
                    # Re-raise other RuntimeErrors
                    raise
            except Exception as e:
                raise e

        else:
            print("No checkpoint found, starting from scratch.")

    start_time = time.time()
    
    for epoch in range(start_epoch, num_epochs+1):
        if max_hours != float('inf') and time.time() - start_time > max_hours * 3600:
            print(f"Maximum hours ({max_hours}) exceeded. Stopping training.")
            break
        _, _, _, epoch_loss = one_epoch(model, optimizer, train_loader, neg_sampler, epoch, loss_fn, loss_type, margin, reg_type, reg_entity_weight, reg_relation_weight)
        log_dict = {'epoch': epoch, 'epoch_loss': epoch_loss}
        _, _, _, eval_loss = one_epoch(model, optimizer, valid_loader, valid_neg_sampler, epoch, loss_fn, loss_type, margin, reg_type, reg_entity_weight, reg_relation_weight, is_train=False)
        log_dict['valid_loss'] = eval_loss
        
        if epoch % eval_every == 0 and evaluator is not None:
            val_metrics = evaluator(model, valid_triples, entity2id, relation2id)
            
            pess = val_metrics.get('pessimistic', {})
            for k, v in pess.items():
                log_dict[f'val_pess_{k}'] = v
                
            # Get MRR for early stopping
            mrr_for_stop = pess.get('MRR', 0.0)
            history.append(val_metrics)
            
            # Log MRR to wandb
            log_dict['val_MRR'] = mrr_for_stop
            
            if mrr_for_stop > best_metric + early_stop_delta:
                best_metric = mrr_for_stop
                best_epoch = epoch
                # Save the best model state
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            # Check for early stopping
            if epoch - best_epoch >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}. Best {early_stop_metric}: {best_metric:.4f}")
                # Restore the best model weights
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break
        
        # Log training metrics
        if log_to_wandb:
            if wandb_run is not None:
                wandb_run.log(log_dict, step=epoch)
            else:
                wandb.log(log_dict, step=epoch)
                
        print(f"Epoch {epoch} | train_loss: {epoch_loss:.6f} | valid_loss: {eval_loss:.6f} | Best {early_stop_metric}: {best_metric:.6f} at epoch {best_epoch}")

        # Save checkpoint atomically
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
        tmp_checkpoint_path = checkpoint_path + '.tmp'
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_metric': best_metric,
            'best_epoch': best_epoch,
            'best_model_state': best_model_state,
            'history': history,
            'torch_rng_state': torch.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'train_loader_generator_state': train_generator.get_state(),
            'neg_sampler_state': neg_sampler.get_state(),
            'model_forward_count': model.forward_count,  # Save forward count for reproducibility
        }
        
        # Save CUDA RNG state if CUDA is available and being used
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            checkpoint_data['cuda_rng_state'] = torch.cuda.get_rng_state()
            # Save RNG state for all CUDA devices if multiple GPUs
            if torch.cuda.device_count() > 1:
                checkpoint_data['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save checkpoint with retry mechanism (up to 5 attempts)
        max_retries = 5
        checkpoint_saved = False
        
        for attempt in range(max_retries):
            try:
                # Clean up any existing temporary file from previous attempts
                if os.path.exists(tmp_checkpoint_path):
                    os.remove(tmp_checkpoint_path)
                
                torch.save(checkpoint_data, tmp_checkpoint_path)
                # Verify the temporary file was created successfully
                if not os.path.exists(tmp_checkpoint_path):
                    raise FileNotFoundError(f"Temporary checkpoint file {tmp_checkpoint_path} was not created")
                
                # Atomic move to final location
                os.replace(tmp_checkpoint_path, checkpoint_path)
                print(f"Checkpoint saved successfully at epoch {epoch} (attempt {attempt + 1})")
                checkpoint_saved = True
                break
                
            except Exception as e:
                print(f"Error saving checkpoint at epoch {epoch} (attempt {attempt + 1}/{max_retries}): {e}")
                # Clean up temporary file if it exists
                if os.path.exists(tmp_checkpoint_path):
                    try:
                        os.remove(tmp_checkpoint_path)
                    except:
                        pass
                
                # If this was the last attempt, give up
                if attempt == max_retries - 1:
                    print("Failed to save checkpoint after all retry attempts. Continuing training without checkpoint...")
                else:
                    print(f"Retrying checkpoint save in 20 seconds...")
                    time.sleep(20)  # Brief pause before retry
    
    # If we never had a validation step (eval_every > num_epochs), save the final model
    if best_model_state is None:
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    # Restore the best model weights
    model.load_state_dict(best_model_state)
    
    return model, history


def train_kge(model, train_triples, valid_triples, entity2id, relation2id, seed_neg, seed_order, seed_init, seed_forward, num_neg_h, num_neg_t, batch_size=128, lr=0.001, max_epochs=1000, eval_every=5, early_stop_metric="MRR", early_stop_patience=15, early_stop_delta=1e-3, evaluator=None, sampler_type="reproducible_on_the_fly", loss_type='bce', margin=None, reg_type='none', reg_entity_weight=0.0, reg_relation_weight=0.0, log_to_wandb=True, wandb_run=None, max_hours=float('inf'), resume_checkpoint=False, checkpoint_dir='.'):
    
    model, optimizer, train_loader, valid_loader, neg_sampler, valid_neg_sampler, train_generator, valid_generator = init_training(
        model, train_triples, valid_triples, entity2id, relation2id, seed_neg, seed_order, seed_init, num_neg_h, num_neg_t, batch_size, lr, sampler_type
    )
    loss_fn = get_loss_fn(loss_type, margin)
    return train_epoch_loop(
        model, optimizer, train_loader, valid_loader, neg_sampler, valid_neg_sampler, max_epochs, valid_triples, entity2id, relation2id, eval_every, early_stop_metric, early_stop_patience, early_stop_delta, evaluator,
        train_generator=train_generator, valid_generator=valid_generator,
        loss_fn=loss_fn, loss_type=loss_type, margin=margin, reg_type=reg_type, reg_entity_weight=reg_entity_weight, reg_relation_weight=reg_relation_weight,
        log_to_wandb=log_to_wandb, wandb_run=wandb_run, max_hours=max_hours,
        resume_checkpoint=resume_checkpoint, checkpoint_dir=checkpoint_dir
    )


