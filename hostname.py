import wandb
import socket
import os
import json
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any

api = wandb.Api()
sites = ["sophia", "nancy", "rennes"]

def extract_hostname(run):
    """
    Essaye plusieurs champs possibles pour récupérer le hostname.
    Retourne une string normalisée (lower()) ou None.
    """
    # 1) metadata (souvent where WANDB stores host)
    try:
        meta = run.metadata or {}
    except Exception:
        meta = {}
    if isinstance(meta, dict):
        for k in ("host", "hostname", "host_name", "HostName"):
            v = meta.get(k)
            if v:
                return str(v).lower()

    # 2) attribut direct (si présent)
    host_attr = getattr(run, "host", None)
    if host_attr:
        return str(host_attr).lower()

    # 3) system_metrics (automatically logged)
    sysm = getattr(run, "system_metrics", None) or {}
    if isinstance(sysm, dict):
        for k in ("host", "hostname", "host_name"):
            v = sysm.get(k)
            if v:
                return str(v).lower()

    # 4) config / summary (fallback, rarement utilisé)
    cfg = getattr(run, "config", {}) or {}
    for k in ("hostname", "host", "HostName"):
        v = cfg.get(k)
        if v:
            return str(v).lower()
    summ = getattr(run, "summary", {}) or {}
    for k in ("hostname", "host"):
        v = summ.get(k)
        if v:
            return str(v).lower()

    # 5) dernier recours : inspecter attrs/_attrs (debug)
    if hasattr(run, "_attrs") and isinstance(run._attrs, dict):
        for k in ("host", "hostname", "metadata"):
            v = run._attrs.get(k)
            if v:
                return str(v).lower()

    return None


# -----------------------------------------------------------------------------
# Legacy: simple prefix-based scan of W&B projects
# -----------------------------------------------------------------------------

def analyze_projects(prefix: str = "28_07"):
    """Backward-compatible helper kept for quick debugging."""
    projects = [p for p in api.projects() if p.name.startswith(prefix)]
    if not projects:
        print(f"Aucun projet commençant par '{prefix}' trouvé.")
        return

    for proj in projects:
        print(f"\n📂 Projet: {proj.name}  (entity={proj.entity})")
        runs = api.runs(f"{proj.entity}/{proj.name}")
        places = defaultdict(list)
        unknown_runs = []

        for run in runs:
            host = extract_hostname(run)
            if host:
                matched = False
                for s in sites:
                    if s in host:
                        places[s].append((run.id, host))
                        matched = True
                        break
                if not matched:
                    places["other"].append((run.id, host))
            else:
                unknown_runs.append(run.id)

        # print summary
        if places:
            for place, rlist in places.items():
                print(f"   → {place}: {len(rlist)} run(s)")
                for rid, h in rlist[:10]:   # affiche max 10 exemples
                    print(f"       - {rid}: {h}")
        else:
            print("   → Aucun host trouvé pour ce projet.")

        if unknown_runs:
            print(f"   → {len(unknown_runs)} run(s) sans host (exemples): {unknown_runs[:10]}")


# -----------------------------------------------------------------------------
# New API – analyse des groupes de runs construits localement
# -----------------------------------------------------------------------------

def _load_wandb_run(run_dir: str):
    """Retourne l'objet ``wandb.Run`` associé à *run_dir* via l'API.

    On lit le fichier ``wandb-metadata.json`` présent dans *run_dir* pour
    récupérer *entity*, *project* et *id*.
    Retourne ``None`` si le run ne peut pas être trouvé.
    """
    meta_file = os.path.join(run_dir, "wandb-metadata.json")
    if not os.path.exists(meta_file):
        print(f"[WARN] {meta_file} manquant – impossible de récupérer le run W&B.")
        return None
    try:
        with open(meta_file, "r") as f:
            meta = json.load(f)
        entity = meta.get("entity")
        project = meta.get("project")
        run_id = meta.get("id")
        if not (entity and project and run_id):
            raise ValueError("Champs entity/project/id manquants dans wandb-metadata.json")
        return api.run(f"{entity}/{project}/{run_id}")
    except Exception as e:
        print(f"[ERR] Impossible d'ouvrir {meta_file}: {e}")
        return None


def analyze_run_groups(all_runs_data: Dict[str, Any]):
    """Analyse les hostnames & états pour chaque groupe de *all_runs_data*.

    Args
    ----
    all_runs_data: structure imbriquée créée dans *stability_measures.py*  :
        model -> quality -> seed_type -> (epoch | groupe spécial) -> List[run_dict]
    """
    GREEN = "\U0001F7E2"   # 🟢
    ORANGE = "\U0001F7E7"  # 🟧
    RED = "\U0001F534"     # 🔴

    for model, quality_data in all_runs_data.items():
        for quality, seed_type_data in quality_data.items():
            for seed_type, epochs_data in seed_type_data.items():
                # epochs_data peut être un dict (with epochs) ou une liste (groupes spéciaux)
                groups_iter = (
                    epochs_data.items() if isinstance(epochs_data, dict) else
                    [("special", epochs_data)]
                )
                for epoch_or_tag, runs in groups_iter:
                    # --- Collect hostnames & states ---
                    hostnames = []
                    states = []
                    created_times = []
                    for run in runs:
                        wb_run = _load_wandb_run(run["run_dir"])
                        if not wb_run:
                            continue
                        hostnames.append(extract_hostname(wb_run) or "unknown")
                        states.append(wb_run.state.lower())
                        # created_at peut être None sur vieilles versions
                        ts = getattr(wb_run, "created_at", None)
                        if ts is None:
                            ts = getattr(wb_run, "_attrs", {}).get("created_at")
                        created_times.append(ts)

                    if not hostnames:
                        print(f"[WARN] Aucun hostname trouvé pour {model}/{quality}/{seed_type}/{epoch_or_tag}")
                        continue

                    # --- Hostname consistency ---
                    same_host = len(set(hostnames)) == 1
                    host_icon = GREEN if same_host else ORANGE

                    # --- Status consistency ---
                    red_warning = False
                    if created_times:
                        # run le plus récent
                        try:
                            latest_idx = max(range(len(created_times)), key=lambda i: created_times[i])
                            if states[latest_idx] != "finished":
                                red_warning = True
                        except Exception:
                            pass
                    # any run not finished
                    if any(s != "finished" for s in states):
                        red_warning = True

                    status_icon = RED if red_warning else ""

                    print(
                        f"{host_icon} {model} | {quality} | {seed_type} | {epoch_or_tag}  "
                        f"→ hosts: {sorted(set(hostnames))}  {'(' + status_icon + ' issues)' if status_icon else ''}"
                    )


# -----------------------------------------------------------------------------
# Execution helper when called as a script
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, pickle, sys

    parser = argparse.ArgumentParser(description="Analyse des hostnames/states pour des groupes de runs.")
    parser.add_argument("--all_runs_data_pkl", type=str, help="Fichier pickle contenant all_runs_data.")
    parser.add_argument("--legacy_prefix", type=str, default=None, help="Analyse rapide par préfixe (mode legacy).")
    args = parser.parse_args()

    if args.legacy_prefix:
        analyze_projects(args.legacy_prefix)
        sys.exit(0)

    if not os.path.exists(args.all_runs_data_pkl):
        print(f"[ERR] Fichier {args.all_runs_data_pkl} introuvable")
        sys.exit(1)

    with open(args.all_runs_data_pkl, "rb") as f:
        all_runs_data = pickle.load(f)

    analyze_run_groups(all_runs_data)
