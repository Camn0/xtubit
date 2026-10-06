from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass

@dataclass
class RestrettoConfig:
    innerbox=(2.0,2.0,2.0)
    outerbox=(8.0,8.0,8.0)
    box_center=(0.0,0.0,0.0)
    search_pitch=1.0
    scoring_pitch=0.25
    memory_size=4096
    receptor="receptor.pdbqt"
    ligand="fragment.mol2"
    output="placements.sdf"
    grid_folder="grid"

    def render(self):
        return "\n".join([
            f"INNERBOX {self.innerbox[0]},{self.innerbox[1]},{self.innerbox[2]}",
            f"OUTERBOX {self.outerbox[0]},{self.outerbox[1]},{self.outerbox[2]}",
            f"BOX_CENTER {self.box_center[0]},{self.box_center[1]},{self.box_center[2]}",
            f"SEARCH_PITCH {self.search_pitch}",
            f"SCORING_PITCH {self.scoring_pitch}",
            f"MEMORY_SIZE {self.memory_size}",
            f"RECEPTOR {self.receptor}",
            f"LIGAND {self.ligand}",
            f"OUTPUT {self.output}",
            f"GRID_FOLDER {self.grid_folder}",
        ])


def write_config(path: str|Path, cfg: RestrettoConfig):
    Path(path).write_text(cfg.render()+"\\n")
