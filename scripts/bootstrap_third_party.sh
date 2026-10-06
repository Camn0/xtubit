#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p third_party
clone_or_update() {
  local url="$1" dir="$2"
  if [ -d "third_party/$dir/.git" ]; then
    git -C "third_party/$dir" fetch --all --tags
  else
    git clone "$url" "third_party/$dir"
  fi
  printf '%s\t%s\n' "$dir" "$(git -C "third_party/$dir" rev-parse HEAD)"
}
: > third_party/COMMITS.txt
clone_or_update https://github.com/bqth29/simulated-bifurcation-algorithm.git simulated-bifurcation-algorithm >> third_party/COMMITS.txt
clone_or_update https://github.com/Tao-qubit/Tabu-Enhanced-Simulated-Bifurcation.git Tabu-Enhanced-Simulated-Bifurcation >> third_party/COMMITS.txt
clone_or_update https://github.com/nonizawa/pSA.git pSA >> third_party/COMMITS.txt
clone_or_update https://github.com/nonizawa/GPU-pSAv.git GPU-pSAv >> third_party/COMMITS.txt
clone_or_update https://github.com/PaulsonLab/Generative_MOBO_qPMHI.git Generative_MOBO_qPMHI >> third_party/COMMITS.txt
clone_or_update https://github.com/vict0rsch/faenet.git faenet >> third_party/COMMITS.txt
clone_or_update https://github.com/RolnickLab/ocp.git ocp >> third_party/COMMITS.txt
clone_or_update https://github.com/akiyamalab/restretto.git restretto >> third_party/COMMITS.txt
clone_or_update https://github.com/akiyamalab/coffee-presc.git coffee-presc >> third_party/COMMITS.txt
clone_or_update https://github.com/alebeneventi/Pi-Stacking.git Pi-Stacking >> third_party/COMMITS.txt
