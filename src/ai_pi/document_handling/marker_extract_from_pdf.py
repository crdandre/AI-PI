"""
1. Uses Marker to extract a formatted markdown from a pdf
2. Uses LLM calls to multimodal models to correct figure caption
fragments that get included in figure images

Steps:
1. Marker conversion
2. with this markdown text, for each image, check the text below, and determine whether it's a complete, partial, or absent caption
3. Do nothing, combine the caption text extracted from the image, or insert the whole caption extracted from the image depending on the case
"""
import logging
import os
import subprocess
import re
import dspy
import json
import time

from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

from dspy_workflow_builder.parse_lm_config import LMForTask, TaskConfig, LMConfig
from dspy_workflow_builder.utils.logging import setup_logging

load_dotenv()


# Create signatures for image analysis
class ImageCaptionExtractor(dspy.Signature):
    image: dspy.Image = dspy.InputField(desc="The image to analyze")
    question: str = dspy.InputField(desc="Question about text in the image")
    answer: str = dspy.OutputField(desc="Extracted text from the image")


class CaptionAnalyzer(dspy.Signature):
    """Analyze text to determine if it contains a figure caption"""
    text: str = dspy.InputField(desc="""Text following an image to analyze. 
        Identify if this is:
        1. A complete figure caption (starts with 'Figure X:' or similar)
        2. A partial caption, which includes:
           - Detailed descriptions of figure parts
           - Anatomical or technical explanations
           - Lettered/numbered subfigure descriptions
           - Text in italics that describes image content
        3. Non-caption text (unrelated to the image)
        
        Consider:
        - Formatting (italics, bold)
        - Technical/anatomical terminology
        - Presence of measurements or part numbers
        - Position immediately after image
        - Descriptive language patterns""")
    answer: str = dspy.OutputField(desc="""String containing JSON response with:
        is_caption (bool): True if text is a complete standalone caption
        is_fragment (bool): True if text is part of a caption or supplementary description
        caption_type (str): "complete", "partial", or "none"
        confidence (float): 0-1 confidence in classification
        cleaned_text (str): Text with formatting preserved""")
    
    
class CaptionCombiner(dspy.Signature):
    """Combine image caption and text fragment into a complete caption while preserving formatting"""
    image_caption: str = dspy.InputField(desc="Caption extracted from image")
    text_fragment: str = dspy.InputField(desc="Caption fragment from text")
    answer: str = dspy.OutputField(desc="""Combined complete caption that:
        1. Preserves any existing formatting (bold, italics, etc.)
        2. Maintains figure numbering if present
        3. Combines information from both sources without redundancy
        4. Uses the formatting style from text_fragment if present""")


class MarkdownSegmenter(dspy.Signature):
    """Determine if text belongs to an image caption"""
    text_block: str = dspy.InputField(desc="Block of text to analyze")
    answer: str = dspy.OutputField(desc="""JSON response with:
        is_caption_content (bool): True if text appears to be part of a caption
        ends_at_line (int): Line number where caption appears to end (0-based)
        confidence (float): 0-1 confidence in assessment""")



class PDFTextExtractor:
    def __init__(
        self,
        output_folder: str = None,
        format: str = "markdown",
        image_caption_lm=None,
        caption_analysis_lm=None,
        caption_combination_lm=None,
        markdown_segmentation_lm=None
    ):
        # Create logger inside the class
        self.logger = logging.getLogger('pdf_extractor')
        self.logger.info("Initializing PDFTextExtractor")
        
        self.output_folder = output_folder
        self.format = format
        
        # Task-specific LMs using enum configurations
        self.image_caption_lm = LMForTask.IMAGE_CAPTION_EXTRACTION.get_lm() if image_caption_lm is None else (
            image_caption_lm.create_lm() if isinstance(image_caption_lm, (LMConfig, TaskConfig)) else image_caption_lm
        )
        self.caption_analysis_lm = LMForTask.CAPTION_ANALYSIS.get_lm() if caption_analysis_lm is None else (
            caption_analysis_lm.create_lm() if isinstance(caption_analysis_lm, (LMConfig, TaskConfig)) else caption_analysis_lm
        )
        self.caption_combination_lm = LMForTask.CAPTION_COMBINATION.get_lm() if caption_combination_lm is None else (
            caption_combination_lm.create_lm() if isinstance(caption_combination_lm, (LMConfig, TaskConfig)) else caption_combination_lm
        )
        self.markdown_segmentation_lm = LMForTask.MARKDOWN_SEGMENTATION.get_lm() if markdown_segmentation_lm is None else (
            markdown_segmentation_lm.create_lm() if isinstance(markdown_segmentation_lm, (LMConfig, TaskConfig)) else markdown_segmentation_lm
        )
        
        self.caption_status = {}

    def extract_pdf(self, input_pdf_path: str, torch_device_for_marker_pdf: str = "cuda:0") -> str:
        """Extract text from a single PDF file and convert to markdown using LLM."""
        if not input_pdf_path or not os.path.exists(input_pdf_path):
            self.logger.error(f"Invalid input PDF path: {input_pdf_path}")
            return None
        
        filename = os.path.basename(input_pdf_path)
        base_name = os.path.splitext(filename)[0]
        # Use self.output_folder if provided, otherwise use input file's directory
        output_dir = self.output_folder if self.output_folder else os.path.dirname(input_pdf_path)
        # Add the subdirectory that Marker creates
        output_subdir = os.path.join(output_dir, base_name)
        output_file = os.path.join(output_subdir, f"{base_name}.md")

        self.logger.debug(f"Input PDF path: {input_pdf_path}")
        self.logger.debug(f"Output directory: {output_dir}")
        self.logger.debug(f"Output subdirectory: {output_subdir}")
        self.logger.debug(f"Expected output file: {output_file}")
        os.makedirs(output_dir, exist_ok=True)

        env = os.environ.copy()
        env["TORCH_DEVICE"] = torch_device_for_marker_pdf
        
        command = [
            "marker_single",
            input_pdf_path,
            "--output_dir",
            output_dir,
            "--use_llm",
            "--output_format",
            self.format
        ]

        try:
            self.logger.info(f"Running command: {' '.join(command)}")
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                env=env
            )
            self.logger.info(f"Marker extraction completed for {input_pdf_path}")
            self.logger.info(f"Marker output: {result.stdout}")
            
            # Wait briefly to ensure file is written
            time.sleep(1)
            
            # Check if file exists in the expected location
            if os.path.exists(output_file):
                return output_file
            
            # If not found in expected location, search in output directory
            self.logger.warning(f"Expected output file not found at {output_file}")
            for root, _, files in os.walk(output_dir):
                for file in files:
                    if file.endswith('.md'):
                        found_file = os.path.join(root, file)
                        self.logger.info(f"Found markdown file at: {found_file}")
                        return found_file
                    
            self.logger.error("No markdown file found in output directory")
            return None
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running marker_single: {e}")
            self.logger.error(f"Marker stderr: {e.stderr}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            self.logger.error(f"Current working directory: {os.getcwd()}")
            return None
    
    
    def _correct_image_figure_segmentation(self, text: str) -> str:
        """
        Process markdown text to handle image captions while preserving all other content.
        """
        lines = text.split('\n')
        result = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # If not an image, keep line and continue
            if not re.search(r'!\[\]\(_page_\d+_Figure_\d+\.jpeg\)', line):
                result.append(line)
                i += 1
                continue
            
            # Found an image - process it and its caption
            result.append(line)  # Keep the image reference
            image_path = re.search(r'!\[\]\((.*?)\)', line).group(1)
            full_image_path = os.path.join(self.output_folder, image_path) if self.output_folder else image_path
            
            # Get next line (potential caption)
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            
            # Analyze the next line for caption content
            analyzer = getattr(dspy, LMForTask.CAPTION_ANALYSIS.get_predictor_type().value)(
                CaptionAnalyzer, 
                lm=self.caption_analysis_lm
            )
            analysis_string = analyzer(text=next_line).answer
            analysis = json.loads(analysis_string)
            
            if analysis['is_caption'] and not analysis['is_fragment']:
                # Complete caption exists - keep it as is
                result.append(next_line)
                i += 2  # Skip past image and caption
            else:
                # Extract caption from image using configured predictor type
                extractor = getattr(dspy, LMForTask.IMAGE_CAPTION_EXTRACTION.get_predictor_type().value)(
                    ImageCaptionExtractor, 
                    lm=self.image_caption_lm
                )
                image_caption = extractor(
                    image=dspy.Image.from_file(full_image_path),
                    question="Extract any figure caption text from this image."
                ).answer.strip()
                
                if analysis['is_fragment']:
                    # Combine partial caption with extracted
                    combined = self.combine_captions(next_line, image_caption)
                    result.append(combined)
                    i += 2  # Skip past image and partial caption
                else:
                    # No caption - insert extracted
                    if image_caption:
                        result.append(image_caption)
                    i += 1  # Skip past just the image
                
        return '\n'.join(result)

    def combine_captions(self, original_text: str, new_text: str) -> str:
        """Combine original and new caption text, preserving italics if present."""
        # Check if the original text is italicized
        is_italicized = original_text.startswith('*') and original_text.endswith('*')
        
        # Remove existing italics markers for clean combination
        if is_italicized:
            original_text = original_text.strip('*')
        
        # Combine the texts
        combined_text = f"{original_text} {new_text}".strip()
        
        # Wrap in italics if the original was italicized
        if is_italicized:
            combined_text = f"*{combined_text}*"
        
        return combined_text


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    # Setup logging using the centralized configuration
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    logger = setup_logging(log_dir, timestamp, "pdf_extractor")
    
    filename = "mmapis.pdf"
    pdf_path = f"/home/christian/projects/agents/ai_pi/examples/{filename}"
    output_folder = f"/home/christian/projects/agents/ai_pi/examples/mmapis"
    
    extractor = PDFTextExtractor(
        output_folder=output_folder,
        format="markdown"
    )
    output_path = extractor.extract_pdf(pdf_path)
    
    logger.info(f"Output path: {output_path}")
    
    