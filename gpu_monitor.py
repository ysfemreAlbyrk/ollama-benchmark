import subprocess
import shutil
from typing import Dict, Optional

class GPUMonitor:
    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index
        self.nvidia_smi_available = shutil.which("nvidia-smi") is not None

    def is_available(self) -> bool:
        """Checks if nvidia-smi is available in the system path."""
        return self.nvidia_smi_available

    def get_gpu_info(self) -> Optional[Dict[str, any]]:
        """
        Queries nvidia-smi for the selected GPU and returns metrics.
        Returns:
            Dict containing:
                - name: GPU model name (str)
                - total_vram: total memory in MiB (int)
                - used_vram: used memory in MiB (int)
                - free_vram: free memory in MiB (int)
        """
        if not self.nvidia_smi_available:
            return None

        try:
            # Construct command to query GPU details
            cmd = [
                "nvidia-smi",
                f"--id={self.gpu_index}",
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits"
            ]
            
            # Execute command
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            output = result.stdout.strip()
            
            if not output:
                return None
                
            # Parse CSV: name, total, used, free
            parts = [p.strip() for p in output.split(",")]
            if len(parts) >= 4:
                return {
                    "name": parts[0],
                    "total_vram": int(parts[1]),
                    "used_vram": int(parts[2]),
                    "free_vram": int(parts[3])
                }
        except (subprocess.SubprocessError, ValueError, IndexError):
            pass
            
        return None

    def get_used_vram(self) -> int:
        """Returns the currently used GPU VRAM in MiB, or 0 if query fails."""
        info = self.get_gpu_info()
        return info["used_vram"] if info else 0
