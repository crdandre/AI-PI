import logging
import os
import json
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Union

class PDFFigures2:
    """A class to extract figures from PDFs using PDFFigures2."""
    
    def __init__(self, jar_path: Optional[str] = None):
        """
        Initialize PDFFigures2 extractor.
        
        Args:
            jar_path: Path to pdffigures2 JAR file. If None, will try to find it in default locations.
        """
        self.jar_path = jar_path
        if not jar_path:
            # Reference implementation from MMAPIS
            dir_path = os.path.dirname(os.path.abspath(__file__))
            self.jar_path = os.path.join(
                dir_path, "pdffigures2", "pdffigures2-assembly-0.0.12-SNAPSHOT.jar"
            )
        
        if not os.path.exists(self.jar_path):
            logging.warning(
                "PDFFigures2 JAR not found at %s. Please provide correct path.", self.jar_path
            )

    def extract_figures(
        self,
        pdf_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        dpi: int = 150,
        timeout: int = 30,
        include_regionless: bool = False,
        verbose: bool = False
    ) -> Optional[Dict]:
        """
        Extract figures from a PDF file.
        
        Args:
            pdf_path: Path to input PDF file
            output_dir: Directory to save output files. If None, uses temp directory
            dpi: DPI for extracted figures (default: 150)
            timeout: Timeout in seconds (default: 30)
            include_regionless: Include captions without figure regions
            verbose: Print verbose output
            
        Returns:
            Dict containing extracted figure data if successful, None otherwise
        """
        if not os.path.exists(pdf_path):
            logging.error("PDF file not found: %s", pdf_path)
            return None

        # Create temp dir if no output dir specified
        temp_dir = None
        if output_dir is None:
            temp_dir = tempfile.mkdtemp()
            output_dir = temp_dir
        else:
            os.makedirs(output_dir, exist_ok=True)

        try:
            # Build command
            cmd = [
                "java",
                "-jar",
                str(self.jar_path),
                str(pdf_path),
                "-d", str(output_dir),  # Save JSON data
                "-m", str(output_dir),  # Save figure images
                "-i", str(dpi)         # Set DPI
            ]
            
            if include_regionless:
                cmd.append("-c")
            
            if not verbose:
                cmd.append("-q")

            # Run PDFFigures2
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE if not verbose else None,
                stderr=subprocess.PIPE if not verbose else None,
                timeout=timeout
            )

            if process.returncode != 0:
                logging.error("Failed to extract figures from %s", pdf_path)
                return None

            if process.stderr:
                logging.warning(f"PDFFigures2 stderr: {process.stderr.decode()}")
            if process.stdout and verbose:
                print(f"PDFFigures2 stdout: {process.stdout.decode()}")

            # Load and return JSON data
            json_path = os.path.join(
                output_dir,
                os.path.basename(pdf_path).replace(".pdf", ".json")
            )
            
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                logging.error("No output JSON found for %s", pdf_path)
                return None

        except subprocess.TimeoutExpired:
            logging.error(
                "PDFFigures2 timed out after %d seconds processing %s",
                timeout,
                pdf_path
            )
            return None
            
        except Exception as e:
            logging.error("Error extracting figures: %s", str(e))
            return None
            
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Static configuration
    INPUT_PDF = "/home/christian/projects/agents/ai_pi/examples/testwocomments.pdf"
    OUTPUT_DIR = "examples/figures"     # Output directory for figures
    JAR_PATH = '/home/christian/projects/agents/ai_pi/src/ai_pi/documents/pdffigures2-assembly-0.0.12-SNAPSHOT.jar'
    DPI = 150                 # Resolution for extracted figures
    TIMEOUT = 30              # Timeout in seconds
    VERBOSE = True            # Print detailed output

    # Initialize extractor
    try:
        extractor = PDFFigures2(jar_path=JAR_PATH)
        
        # Extract figures
        figures_data = extractor.extract_figures(
            pdf_path=INPUT_PDF,
            output_dir=OUTPUT_DIR,
            dpi=DPI,
            timeout=TIMEOUT,
            include_regionless=True,
            verbose=VERBOSE
        )
        
        if figures_data:
            print(f"\nSuccessfully extracted {len(figures_data)} figures:")
            for i, figure in enumerate(figures_data, 1):
                caption = figure.get('caption', 'No caption available')
                print(f"\nFigure {i}:")
                print(f"Caption: {caption[:200]}...")  # Print first 200 chars of caption
                print(f"Page: {figure.get('page', 'Unknown')}")
                if VERBOSE:
                    print(f"Boundary: {figure.get('regionBoundary', 'No boundary info')}")
        else:
            print("No figures were extracted from the PDF")
            sys.exit(1)

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
