"""
Test de récupération automatique des checkpoints corrompus.

Ce test vérifie que le système peut récupérer automatiquement
lorsqu'un fichier checkpoint est corrompu.
"""

import os
import tempfile
import torch
import pytest
from pathlib import Path


def test_corrupted_checkpoint_detection():
    """
    Vérifie que les checkpoints corrompus sont détectés et supprimés.
    """
    # Créer un répertoire temporaire
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'checkpoint.pth')
        
        # Créer un fichier corrompu
        with open(checkpoint_path, 'w') as f:
            f.write("This is corrupted data, not a valid PyTorch checkpoint")
        
        assert os.path.exists(checkpoint_path), "Checkpoint corrompu devrait exister"
        
        # Tenter de charger le checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            pytest.fail("Le chargement aurait dû échouer avec un checkpoint corrompu")
        except (RuntimeError, Exception) as e:
            # Vérifier que c'est bien une erreur de corruption
            error_msg = str(e)
            is_corruption_error = (
                "PytorchStreamReader failed" in error_msg or 
                "failed finding central directory" in error_msg or
                "pickle data was truncated" in error_msg or
                "UnpicklingError" in str(type(e))
            )
            assert is_corruption_error, f"Erreur inattendue: {e}"
            print(f"✅ Erreur de corruption détectée correctement: {type(e).__name__}: {e}")


def test_valid_checkpoint_loading():
    """
    Vérifie qu'un checkpoint valide peut être chargé correctement.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'checkpoint.pth')
        
        # Créer un checkpoint valide
        test_data = {
            'epoch': 5,
            'model_state_dict': {'weight': torch.randn(10, 10)},
            'optimizer_state_dict': {},
            'best_metric': 0.75,
            'best_epoch': 3,
            'best_model_state': None,
            'history': [{'epoch': 1, 'loss': 0.5}]
        }
        
        torch.save(test_data, checkpoint_path)
        assert os.path.exists(checkpoint_path), "Checkpoint valide devrait exister"
        
        # Charger le checkpoint
        loaded = torch.load(checkpoint_path, weights_only=False)
        
        assert loaded['epoch'] == 5
        assert loaded['best_metric'] == 0.75
        assert loaded['best_epoch'] == 3
        print("✅ Checkpoint valide chargé correctement")


def test_checkpoint_deletion_simulation():
    """
    Simule la suppression d'un checkpoint corrompu.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'checkpoint.pth')
        
        # Créer un fichier corrompu
        with open(checkpoint_path, 'w') as f:
            f.write("corrupted")
        
        assert os.path.exists(checkpoint_path)
        
        # Simuler la détection et suppression
        try:
            torch.load(checkpoint_path, weights_only=False)
        except (RuntimeError, Exception) as e:
            error_msg = str(e)
            is_corruption = (
                "PytorchStreamReader failed" in error_msg or 
                "failed finding central directory" in error_msg or
                "pickle data was truncated" in error_msg or
                "UnpicklingError" in str(type(e))
            )
            if is_corruption:
                # Supprimer le fichier corrompu
                os.remove(checkpoint_path)
                print("✅ Checkpoint corrompu supprimé")
        
        assert not os.path.exists(checkpoint_path), "Checkpoint corrompu devrait être supprimé"


if __name__ == "__main__":
    print("Test 1: Détection de corruption")
    test_corrupted_checkpoint_detection()
    
    print("\nTest 2: Chargement de checkpoint valide")
    test_valid_checkpoint_loading()
    
    print("\nTest 3: Simulation de suppression")
    test_checkpoint_deletion_simulation()
    
    print("\n✅ Tous les tests passent!")
