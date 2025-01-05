"""
1. Uses Marker to extract a formatted markdown from a pdf
2. Uses LLM calls to multimodal models to correct figure caption
fragments that get included in figure images

Steps:
1. Marker conversion
2. with this markdown text, for each image, check the text below, and determine whether it's a complete, partial, or absent caption
3. Do nothing, combine the caption text extracted from the image, or insert the whole caption extracted from the image depending on the case
"""

from dotenv import load_dotenv
load_dotenv()

import logging
import os
import subprocess
import re
import dspy
import json
import time


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
        lm,
        output_folder: str = None,
        format: str = "markdown"
    ):
        logging.info("Initializing PDFTextExtractor")
        self.output_folder = output_folder
        self.format = format
        self.lm = lm
        self.caption_status = {}  # Track caption status for each image

    def extract_pdf(self, input_pdf_path: str) -> str:
        """Extract text from a single PDF file and convert to markdown using LLM."""
        if not input_pdf_path or not os.path.exists(input_pdf_path):
            logging.error(f"Invalid input PDF path: {input_pdf_path}")
            return None
        
        filename = os.path.basename(input_pdf_path)
        base_name = os.path.splitext(filename)[0]
        # Use self.output_folder if provided, otherwise use input file's directory
        output_dir = self.output_folder if self.output_folder else os.path.dirname(input_pdf_path)
        # Add the subdirectory that Marker creates
        output_subdir = os.path.join(output_dir, base_name)
        output_file = os.path.join(output_subdir, f"{base_name}.md")

        # Debug logging
        logging.info(f"Input PDF path: {input_pdf_path}")
        logging.info(f"Output directory: {output_dir}")
        logging.info(f"Output subdirectory: {output_subdir}")
        logging.info(f"Expected output file: {output_file}")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

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
            logging.info(f"Running command: {' '.join(command)}")
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            logging.info(f"Marker extraction completed for {input_pdf_path}")
            logging.info(f"Marker output: {result.stdout}")
            
            # Wait briefly to ensure file is written
            time.sleep(1)
            
            # Check if file exists in the expected location
            if os.path.exists(output_file):
                return output_file
            
            # If not found in expected location, search in output directory
            logging.warning(f"Expected output file not found at {output_file}")
            for root, _, files in os.walk(output_dir):
                for file in files:
                    if file.endswith('.md'):
                        found_file = os.path.join(root, file)
                        logging.info(f"Found markdown file at: {found_file}")
                        return found_file
                    
            logging.error("No markdown file found in output directory")
            return None
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running marker_single: {e}")
            logging.error(f"Marker stderr: {e.stderr}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            logging.error(f"Current working directory: {os.getcwd()}")
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
            analyzer = dspy.Predict(CaptionAnalyzer, lm=self.lm)
            analysis_string = analyzer(text=next_line).answer
            logging.info("analysis_string obtained")
            analysis = json.loads(analysis_string)
            logging.info("analysis json parsed")
            
            if analysis['is_caption'] and not analysis['is_fragment']:
                # Complete caption exists - keep it as is
                result.append(next_line)
                i += 2  # Skip past image and caption
            else:
                # Extract caption from image
                extractor = dspy.Predict(ImageCaptionExtractor, lm=self.lm)
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
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure OpenRouter LLM
    openrouter_model = 'openrouter/openai/gpt-4o-mini'
    lm = dspy.LM(
        openrouter_model,
        api_base="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.01,
    )
    dspy.settings.configure(lm=lm)
    
    filename = "mmapis.pdf"
    pdf_path = f"/home/christian/projects/agents/ai_pi/examples/{filename}"
    output_folder = f"/home/christian/projects/agents/ai_pi/examples/mmapis"
    
    extractor = PDFTextExtractor(
        lm=lm,
        output_folder=output_folder,
        format="markdown"
    )
    output_path = extractor.extract_pdf(pdf_path)
    
    print(output_path)

    #Test _correct_image_figure_segmentation
    
    # filename = "testwocomments"
    # test_markdown_path = f"/home/christian/projects/agents/ai_pi/examples/{filename}/{filename}.md"
    # output_path = test_markdown_path.replace('.md', '_corrected.md')
    
    # extractor = PDFTextExtractor(
    #     lm=lm,
    #     output_folder=f"/home/christian/projects/agents/ai_pi/examples/{filename}/"
    # )
    
    # # Read the test markdown file
    # with open(test_markdown_path, 'r', encoding='utf-8') as f:
    #     test_markdown = f.read()
    
    # # Process the markdown and correct image segmentation
    # corrected_markdown = extractor._correct_image_figure_segmentation(test_markdown)
    
    # # Write corrected markdown to new file
    # with open(output_path, 'w', encoding='utf-8') as f:
    #     f.write(corrected_markdown)
    
    # print(f"Corrected markdown written to: {output_path}")
    
    