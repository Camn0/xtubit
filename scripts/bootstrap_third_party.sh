#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p third_party
clone_or_checkout() {
  local url="$1" dir="$2" sha="$3"
  if [ ! -d "third_party/$dir/.git" ]; then
    git clone "$url" "third_party/$dir"
  fi
  git -C "third_party/$dir" fetch --all --tags --quiet
  git -C "third_party/$dir" checkout --detach "$sha" --quiet
  printf '%s\t%s\n' "$dir" "$(git -C "third_party/$dir" rev-parse HEAD)"
}

: > third_party/COMMITS.txt
clone_or_checkout https://github.com/bqth29/simulated-bifurcation-algorithm.git simulated-bifurcation-algorithm dcff6d6b43b714bc512c19ab846f61d495dfea8f >> third_party/COMMITS.txt
clone_or_checkout https://github.com/Tao-qubit/Tabu-Enhanced-Simulated-Bifurcation.git Tabu-Enhanced-Simulated-Bifurcation e41481fae81c2e3c49841af0b1e27b8c14872fd8 >> third_party/COMMITS.txt
clone_or_checkout https://github.com/nonizawa/pSA.git pSA 551845188d79f4b6a8cc2b2c956927b526f7ed3c >> third_party/COMMITS.txt
clone_or_checkout https://github.com/nonizawa/GPU-pSAv.git GPU-pSAv 61c8e1cf577c3871c9b089f0fc6e8b53851475db >> third_party/COMMITS.txt
clone_or_checkout https://github.com/PaulsonLab/Generative_MOBO_qPMHI.git Generative_MOBO_qPMHI d32f3d1eceee01e2d8c775e5b29b3f5729116d9a >> third_party/COMMITS.txt
clone_or_checkout https://github.com/vict0rsch/faenet.git faenet 1f725cfabb0ef47a44662eac7bc39325249226b2 >> third_party/COMMITS.txt
clone_or_checkout https://github.com/akiyamalab/restretto.git restretto 5d03209c79f0b632c28ef07ca1b07a1eca0fc2be >> third_party/COMMITS.txt
clone_or_checkout https://github.com/akiyamalab/coffee-presc.git coffee-presc 61121c5a5b8452fa5d224872358c1da79ef61598 >> third_party/COMMITS.txt
clone_or_checkout https://github.com/recruit-communications/pyqubo.git pyqubo 717c680dae392e769191f359e41c57445137af5d >> third_party/COMMITS.txt
