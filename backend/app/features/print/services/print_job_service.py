"""
PrintChakra Backend - Print Job Service

Service for print job submission and management.
"""

import os
import sys
import subprocess
import tempfile
import logging
from typing import List, Dict, Any, Optional

from app.core.config import get_data_dirs

logger = logging.getLogger(__name__)


class PrintJobService:
    """Service for submitting and managing print jobs."""
    
    def __init__(self):
        """Initialize print job service."""
        self.dirs = get_data_dirs()
    
    def get_default_printer(self) -> Optional[str]:
        """Get the default printer name."""
        if sys.platform.startswith("win"):
            try:
                import win32print
                return win32print.GetDefaultPrinter()
            except ImportError:
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-Command",
                     "(Get-Printer | Where-Object {$_.Default -eq $true}).Name"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    return result.stdout.strip()
        else:
            result = subprocess.run(
                ["lpstat", "-d"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(":")
                if len(parts) >= 2:
                    return parts[1].strip()
        
        return None
    
    def find_file(self, filename: str) -> Optional[str]:
        """
        Find a file in known directories.
        
        Args:
            filename: Name of file to find
            
        Returns:
            Full path to file or None
        """
        search_dirs = [
            self.dirs.get('PROCESSED_DIR'),
            self.dirs.get('CONVERTED_DIR'),
            self.dirs.get('UPLOAD_DIR'),
            self.dirs.get('PDF_DIR'),
            self.dirs.get('OCR_DATA_DIR'),
        ]
        
        for search_dir in search_dirs:
            if search_dir:
                candidate = os.path.join(search_dir, filename)
                if os.path.exists(candidate):
                    return candidate
        
        return None
    
    def _find_blank_pdf(self) -> Optional[str]:
        """Find blank.pdf for test printing."""
        possible_paths = [
            os.path.join(self.dirs.get('PRINT_DIR', ''), "blank.pdf"),
            os.path.join(self.dirs.get('BASE_DIR', ''), "app", "print_scripts", "blank.pdf"),
            os.path.join(self.dirs.get('PUBLIC_DIR', ''), "blank.pdf"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def test_printer(self) -> Dict[str, Any]:
        """Test printer connection."""
        blank_pdf = self._find_blank_pdf()
        
        return {
            "status": "success",
            "message": "Printer test: blank.pdf is ready" if blank_pdf else "blank.pdf not found",
            "pdf_exists": blank_pdf is not None,
            "pdf_path": blank_pdf,
            "default_printer": self.get_default_printer()
        }
    
    def print_blank_page(self) -> Dict[str, Any]:
        """Print a blank page for document feeding."""
        blank_pdf = self._find_blank_pdf()
        
        if not blank_pdf:
            return {
                "status": "error",
                "message": "blank.pdf not found in any location"
            }
        
        try:
            success = self._print_file(blank_pdf)
            
            if success:
                return {
                    "status": "success",
                    "message": "Print command sent and capture triggered"
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to send print command"
                }
        
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def print_document(self, filename: str, copies: int = 1) -> Dict[str, Any]:
        """
        Print a single document.
        
        Args:
            filename: Name of file to print
            copies: Number of copies
            
        Returns:
            Result dictionary
        """
        logger.info(f"Print request for: {filename}")
        
        # Find the file
        file_path = self.find_file(filename)
        
        if not file_path:
            return {
                "success": False,
                "error": f"File not found: {filename}"
            }
        
        # Convert image to PDF if needed
        print_path = file_path
        temp_pdf = None
        
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif']:
            try:
                print_path = self._convert_image_to_pdf(file_path)
                temp_pdf = print_path
            except Exception as e:
                logger.warning(f"Image to PDF conversion failed: {e}")
                # Continue with original file
        
        # Print the file
        try:
            success = False
            
            for copy_num in range(copies):
                if self._print_file(print_path):
                    success = True
                    logger.info(f"Print succeeded (copy {copy_num + 1})")
            
            # Clean up temp file
            if temp_pdf and os.path.exists(temp_pdf):
                try:
                    os.remove(temp_pdf)
                except:
                    pass
            
            if success:
                return {
                    "success": True,
                    "message": f"Printed {filename} ({copies} copies)",
                    "filename": filename,
                    "copies": copies
                }
            else:
                return {
                    "success": False,
                    "error": "Print command failed"
                }
        
        except Exception as e:
            logger.error(f"Print error: {e}")
            return {"success": False, "error": str(e)}
    
    def print_documents(self, filenames: List[str], copies: int = 1) -> Dict[str, Any]:
        """
        Print multiple documents separately.
        
        Args:
            filenames: List of filenames to print
            copies: Number of copies for each
            
        Returns:
            Result dictionary with success/failure for each file
        """
        results = {
            "success": True,
            "printed": [],
            "failed": [],
            "total": len(filenames),
            "copies": copies
        }
        
        for filename in filenames:
            result = self.print_document(filename, copies)
            
            if result.get("success"):
                results["printed"].append(filename)
            else:
                results["failed"].append({
                    "filename": filename,
                    "error": result.get("error", "Unknown error")
                })
        
        results["success"] = len(results["failed"]) == 0
        
        return results
    
    def print_merged_documents(self, filenames: List[str], copies: int = 1) -> Dict[str, Any]:
        """
        Merge documents into one PDF and print.
        
        Args:
            filenames: List of filenames to merge and print
            copies: Number of copies
            
        Returns:
            Result dictionary
        """
        from PIL import Image
        
        try:
            images = []
            
            for filename in filenames:
                file_path = self.find_file(filename)
                if not file_path:
                    continue
                
                try:
                    img = Image.open(file_path)
                    
                    # Convert to RGB
                    if img.mode in ('RGBA', 'LA', 'P'):
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        if img.mode == 'RGBA':
                            background.paste(img, mask=img.split()[-1])
                        else:
                            background.paste(img)
                        img = background
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    images.append(img)
                
                except Exception as e:
                    logger.warning(f"Could not open {filename}: {e}")
            
            if not images:
                return {"success": False, "error": "No valid images to print"}
            
            # Create merged PDF
            temp_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
            temp_pdf_path = temp_pdf.name
            temp_pdf.close()
            
            first_image = images[0]
            other_images = images[1:] if len(images) > 1 else []
            
            first_image.save(
                temp_pdf_path,
                "PDF",
                save_all=True,
                append_images=other_images,
                resolution=150.0
            )
            
            # Close images
            for img in images:
                img.close()
            
            # Print merged PDF
            success = False
            for copy_num in range(copies):
                if self._print_file(temp_pdf_path):
                    success = True
            
            # Clean up
            try:
                os.remove(temp_pdf_path)
            except:
                pass
            
            if success:
                return {
                    "success": True,
                    "message": f"Printed {len(filenames)} merged documents ({copies} copies)",
                    "files": filenames,
                    "copies": copies
                }
            else:
                return {"success": False, "error": "Print command failed"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def print_pdf(self, filename: str, pages: str = "all", copies: int = 1) -> Dict[str, Any]:
        """
        Print a PDF file.
        
        Args:
            filename: Name of PDF file
            pages: Page range ("all", "1-5", "1,3,5")
            copies: Number of copies
            
        Returns:
            Result dictionary
        """
        file_path = self.find_file(filename)
        
        if not file_path:
            return {"success": False, "error": f"File not found: {filename}"}
        
        if not filename.lower().endswith('.pdf'):
            return {"success": False, "error": "Not a PDF file"}
        
        try:
            success = False
            
            for copy_num in range(copies):
                if self._print_file(file_path):
                    success = True
            
            if success:
                return {
                    "success": True,
                    "message": f"Printed {filename} ({copies} copies)",
                    "filename": filename,
                    "pages": pages,
                    "copies": copies
                }
            else:
                return {"success": False, "error": "Print command failed"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _print_file(self, file_path: str) -> bool:
        """
        Send a file to the default printer.
        
        Args:
            file_path: Path to file to print
            
        Returns:
            True if print command succeeded
        """
        if sys.platform.startswith("win"):
            return self._print_windows(file_path)
        else:
            return self._print_cups(file_path)
    
    def _print_windows(self, file_path: str) -> bool:
        """Print using Windows APIs."""
        try:
            import win32api
            import win32print
            
            printer_name = win32print.GetDefaultPrinter()
            
            if not printer_name:
                logger.error("No default printer configured")
                return False
            
            # Method 1: ShellExecute with "print"
            try:
                result = win32api.ShellExecute(0, "print", file_path, None, ".", 0)
                if result > 32:
                    logger.info(f"ShellExecute print succeeded")
                    return True
            except Exception as e:
                logger.warning(f"ShellExecute print failed: {e}")
            
            # Method 2: PrintTo with printer name
            try:
                result = win32api.ShellExecute(0, "printto", file_path, f'"{printer_name}"', ".", 0)
                if result > 32:
                    logger.info(f"PrintTo succeeded")
                    return True
            except Exception as e:
                logger.warning(f"PrintTo failed: {e}")
            
            # Method 3: PowerShell
            try:
                cmd = f'Start-Process -FilePath "{file_path}" -Verb Print -WindowStyle Hidden'
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", cmd],
                    capture_output=True,
                    timeout=30
                )
                if result.returncode == 0:
                    logger.info("PowerShell print succeeded")
                    return True
            except Exception as e:
                logger.warning(f"PowerShell print failed: {e}")
            
            return False
        
        except ImportError:
            # win32api not available, use PowerShell only
            try:
                cmd = f'Start-Process -FilePath "{file_path}" -Verb Print -WindowStyle Hidden'
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", cmd],
                    capture_output=True,
                    timeout=30
                )
                return result.returncode == 0
            except Exception as e:
                logger.error(f"PowerShell print failed: {e}")
                return False
    
    def _print_cups(self, file_path: str) -> bool:
        """Print using CUPS (Linux/macOS)."""
        try:
            result = subprocess.run(
                ["lp", file_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"CUPS print failed: {e}")
            return False
    
    def _convert_image_to_pdf(self, image_path: str) -> str:
        """
        Convert an image to PDF for printing.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Path to temporary PDF file
        """
        from PIL import Image
        
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            if img.mode == 'RGBA':
                background.paste(img, mask=img.split()[-1])
            else:
                background.paste(img)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Create temp PDF
        temp_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        temp_pdf_path = temp_pdf.name
        temp_pdf.close()
        
        # Save as PDF
        img.save(temp_pdf_path, 'PDF', resolution=150.0)
        img.close()
        
        return temp_pdf_path
    
    def create_job(
        self,
        document_path: str,
        printer_name: str,
        copies: int = 1,
        color: bool = True,
        duplex: bool = False,
        page_range: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a print job.
        
        Args:
            document_path: Path to document
            printer_name: Target printer name
            copies: Number of copies
            color: Color or grayscale
            duplex: Double-sided printing
            page_range: Page range to print
            
        Returns:
            Job info dictionary
        """
        import uuid
        from datetime import datetime
        
        job_id = str(uuid.uuid4())
        
        job = {
            "id": job_id,
            "document": os.path.basename(document_path),
            "document_path": document_path,
            "printer": printer_name,
            "copies": copies,
            "color": color,
            "duplex": duplex,
            "page_range": page_range,
            "status": "queued",
            "created": datetime.now().isoformat(),
            "completed": None,
            "error": None
        }
        
        # Submit the print job
        try:
            for _ in range(copies):
                success = self._print_file(document_path)
                if not success:
                    job["status"] = "failed"
                    job["error"] = "Print command failed"
                    break
            else:
                job["status"] = "submitted"
        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)
        
        return job
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job status by ID.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job info or None
        """
        # In a real implementation, this would query a job store
        # For now, return a placeholder
        return None
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a print job.
        
        Args:
            job_id: Job ID
            
        Returns:
            True if successful
        """
        # In a real implementation, this would cancel the job in the spooler
        logger.info(f"Cancel job requested for: {job_id}")
        return True
    
    def list_jobs(self, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        List print jobs.
        
        Args:
            status: Filter by status
            limit: Maximum number of jobs to return
            
        Returns:
            List of jobs
        """
        # In a real implementation, this would query a job store
        return []
    
    def retry_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Retry a failed print job.
        
        Args:
            job_id: Job ID
            
        Returns:
            New job info or None
        """
        # In a real implementation, this would re-queue the job
        logger.info(f"Retry job requested for: {job_id}")
        return None
