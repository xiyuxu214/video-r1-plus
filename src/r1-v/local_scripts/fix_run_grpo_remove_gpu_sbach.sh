#!/bin/sh
# Run in local_scripts. Safely drops ONLY obsolete GPU #SBATCH lines (any line number).
# Safe on new run_grpo_3gpu_245.sh (no-op if none match). Does not use sed -i on NFS.
set -e
f=run_grpo_3gpu_245.sh
test -f "$f" || { echo "missing $f"; exit 1; }
cp -a "$f" "${f}.bak.$(date +%Y%m%d%H%M%S)"
awk '
  /^#SBATCH --gpus-per-node/ { next }
  /^#SBATCH -G[[:space:]]/    { next }
  /^#SBATCH --gres=/         { next }
  /^## #SBATCH -G[[:space:]]/ { next }
  /^## #SBATCH --gres=/      { next }
  { print }
' "$f" > "${f}.new"
mv "${f}.new" "$f"
echo "OK. #SBATCH lines now:"
grep -n '^#SBATCH' "$f" || true
echo ""
echo "Submit (pick ONE that your site accepts):"
echo "  bash submit_grpo_3gpu_245_ln207.sh"
echo "  yhbatch -G 3 $f"
echo "  yhbatch --gpus-per-node=3 $f"
echo "  yhbatch --gres=dcu:3 $f"
echo "  yhbatch --gres=gpu:3 $f"
