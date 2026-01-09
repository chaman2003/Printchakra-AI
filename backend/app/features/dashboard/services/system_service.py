"""
PrintChakra Backend - System Service

Service for system-related operations including printer management.
"""

import os
import platform
import shutil
import subprocess
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class SystemService:
    """Service for system information and configuration."""
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive system information.
        
        Returns:
            Dictionary containing system details
        """
        return {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "printers": self.get_printers(),
            "default_printer": self.get_default_printer(),
        }
    
    def get_printers(self) -> List[Dict[str, Any]]:
        """
        Get list of available printers.
        
        Returns:
            List of printer information dictionaries
        """
        system = platform.system()
        printers = []
        
        try:
            if system == "Windows":
                printers = self._get_windows_printers()
            elif system == "Darwin":  # macOS
                printers = self._get_macos_printers()
            elif system == "Linux":
                printers = self._get_linux_printers()
        except Exception as e:
            logger.error(f"Error getting printers: {e}")
        
        return printers
    
    def _get_windows_printers(self) -> List[Dict[str, Any]]:
        """Get printers on Windows."""
        printers = []
        try:
            result = subprocess.run(
                ["powershell", "-Command", 
                 "Get-Printer | Select-Object Name, DriverName, PortName, PrinterStatus | ConvertTo-Json"],
                capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0 and result.stdout.strip():
                import json
                data = json.loads(result.stdout)
                if isinstance(data, dict):
                    data = [data]
                for p in data:
                    printers.append({
                        "name": p.get("Name", "Unknown"),
                        "driver": p.get("DriverName", "Unknown"),
                        "port": p.get("PortName", "Unknown"),
                        "status": self._parse_printer_status(p.get("PrinterStatus", 0)),
                    })
        except Exception as e:
            logger.error(f"Error getting Windows printers: {e}")
        
        return printers
    
    def _get_macos_printers(self) -> List[Dict[str, Any]]:
        """Get printers on macOS."""
        printers = []
        try:
            result = subprocess.run(
                ["lpstat", "-p"], capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line.startswith("printer"):
                        parts = line.split()
                        if len(parts) >= 2:
                            printers.append({
                                "name": parts[1],
                                "status": "idle" if "idle" in line.lower() else "busy",
                            })
        except Exception as e:
            logger.error(f"Error getting macOS printers: {e}")
        
        return printers
    
    def _get_linux_printers(self) -> List[Dict[str, Any]]:
        """Get printers on Linux."""
        printers = []
        try:
            result = subprocess.run(
                ["lpstat", "-p"], capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line.startswith("printer"):
                        parts = line.split()
                        if len(parts) >= 2:
                            printers.append({
                                "name": parts[1],
                                "status": "idle" if "idle" in line.lower() else "busy",
                            })
        except Exception as e:
            logger.error(f"Error getting Linux printers: {e}")
        
        return printers
    
    def _parse_printer_status(self, status_code: int) -> str:
        """Parse Windows printer status code."""
        status_map = {
            0: "ready",
            1: "paused",
            2: "error",
            3: "pending_deletion",
            4: "paper_jam",
            5: "paper_out",
            6: "manual_feed",
            7: "paper_problem",
            8: "offline",
            9: "io_active",
            10: "busy",
            11: "printing",
            12: "output_bin_full",
            13: "not_available",
            14: "waiting",
            15: "processing",
            16: "initializing",
            17: "warming_up",
            18: "toner_low",
            19: "no_toner",
            20: "page_punt",
            21: "user_intervention",
            22: "out_of_memory",
            23: "door_open",
            24: "server_unknown",
            25: "power_save",
        }
        return status_map.get(status_code, "unknown")
    
    def get_default_printer(self) -> Optional[str]:
        """
        Get the default printer name.
        
        Returns:
            Default printer name or None
        """
        system = platform.system()
        
        try:
            if system == "Windows":
                result = subprocess.run(
                    ["powershell", "-Command",
                     "(Get-WmiObject -Query 'SELECT * FROM Win32_Printer WHERE Default=TRUE').Name"],
                    capture_output=True, text=True, timeout=15
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
            
            elif system in ["Darwin", "Linux"]:
                result = subprocess.run(
                    ["lpstat", "-d"], capture_output=True, text=True, timeout=15
                )
                if result.returncode == 0:
                    # Parse "system default destination: printer_name"
                    output = result.stdout.strip()
                    if ":" in output:
                        return output.split(":")[-1].strip()
        
        except Exception as e:
            logger.error(f"Error getting default printer: {e}")
        
        return None
    
    def set_default_printer(self, printer_name: str) -> Dict[str, Any]:
        """
        Set the default printer.
        
        Args:
            printer_name: Name of printer to set as default
            
        Returns:
            Result dictionary with success status
        """
        system = platform.system()
        
        try:
            if system == "Windows":
                result = subprocess.run(
                    ["powershell", "-Command",
                     f'(Get-WmiObject -Query "SELECT * FROM Win32_Printer WHERE Name=\'{printer_name}\'").SetDefaultPrinter()'],
                    capture_output=True, text=True, timeout=15
                )
                if result.returncode == 0:
                    return {"success": True, "printer": printer_name}
                else:
                    return {"success": False, "error": result.stderr or "Failed to set default printer"}
            
            elif system in ["Darwin", "Linux"]:
                result = subprocess.run(
                    ["lpoptions", "-d", printer_name],
                    capture_output=True, text=True, timeout=15
                )
                if result.returncode == 0:
                    return {"success": True, "printer": printer_name}
                else:
                    return {"success": False, "error": result.stderr or "Failed to set default printer"}
            
            else:
                return {"success": False, "error": f"Unsupported platform: {system}"}
        
        except Exception as e:
            logger.error(f"Error setting default printer: {e}")
            return {"success": False, "error": str(e)}
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """
        Get GPU information for ML operations.
        
        Returns:
            GPU info dictionary
        """
        gpu_info = {
            "available": False,
            "cuda_available": False,
            "device_count": 0,
            "devices": [],
            "memory_total": 0,
            "memory_used": 0,
            "memory_free": 0
        }
        
        try:
            import torch
            
            gpu_info["cuda_available"] = torch.cuda.is_available()
            
            if torch.cuda.is_available():
                gpu_info["available"] = True
                gpu_info["device_count"] = torch.cuda.device_count()
                
                for i in range(torch.cuda.device_count()):
                    device_props = torch.cuda.get_device_properties(i)
                    memory_total = device_props.total_memory
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_free = memory_total - memory_allocated
                    
                    gpu_info["devices"].append({
                        "index": i,
                        "name": device_props.name,
                        "memory_total": memory_total,
                        "memory_used": memory_allocated,
                        "memory_free": memory_free,
                        "compute_capability": f"{device_props.major}.{device_props.minor}"
                    })
                    
                    if i == 0:
                        gpu_info["memory_total"] = memory_total
                        gpu_info["memory_used"] = memory_allocated
                        gpu_info["memory_free"] = memory_free
        
        except ImportError:
            gpu_info["error"] = "PyTorch not installed"
        except Exception as e:
            gpu_info["error"] = str(e)
        
        return gpu_info
    
    def get_services_status(self) -> Dict[str, Any]:
        """
        Get status of external services.
        
        Returns:
            Services status dictionary
        """
        services = {
            "ollama": {"status": "unknown", "available": False},
            "paddle_ocr": {"status": "unknown", "available": False},
            "print_spooler": {"status": "unknown", "available": False}
        }
        
        # Check Ollama
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                services["ollama"] = {"status": "running", "available": True}
            else:
                services["ollama"] = {"status": "error", "available": False}
        except Exception:
            services["ollama"] = {"status": "not running", "available": False}
        
        # Check PaddleOCR
        try:
            from paddleocr import PaddleOCR
            services["paddle_ocr"] = {"status": "available", "available": True}
        except ImportError:
            services["paddle_ocr"] = {"status": "not installed", "available": False}
        except Exception:
            services["paddle_ocr"] = {"status": "error", "available": False}
        
        # Check Print Spooler (Windows)
        if platform.system() == "Windows":
            try:
                result = subprocess.run(
                    ["powershell", "-Command", "(Get-Service -Name Spooler).Status"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    status = result.stdout.strip().lower()
                    services["print_spooler"] = {
                        "status": status,
                        "available": status == "running"
                    }
            except Exception:
                pass
        
        return services
    
    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get storage information.
        
        Returns:
            Storage info dictionary
        """
        import shutil
        
        storage = {
            "total": 0,
            "used": 0,
            "free": 0,
            "percent_used": 0,
            "data_directories": {}
        }
        
        try:
            # Get disk usage for the current drive
            if platform.system() == "Windows":
                drive = os.path.splitdrive(os.getcwd())[0] + "\\"
            else:
                drive = "/"
            
            total, used, free = shutil.disk_usage(drive)
            storage["total"] = total
            storage["used"] = used
            storage["free"] = free
            storage["percent_used"] = round((used / total) * 100, 2) if total > 0 else 0
            
            # Check data directories if they exist
            data_base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "data")
            
            if os.path.exists(data_base):
                for folder in ["uploads", "processed", "converted", "pdfs"]:
                    folder_path = os.path.join(data_base, folder)
                    if os.path.exists(folder_path):
                        folder_size = 0
                        file_count = 0
                        for dirpath, dirnames, filenames in os.walk(folder_path):
                            for f in filenames:
                                fp = os.path.join(dirpath, f)
                                if os.path.exists(fp):
                                    folder_size += os.path.getsize(fp)
                                    file_count += 1
                        
                        storage["data_directories"][folder] = {
                            "size": folder_size,
                            "file_count": file_count
                        }
        
        except Exception as e:
            storage["error"] = str(e)
        
        return storage
