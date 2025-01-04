"""
1. Uses Marker to extract a formatted markdown from a pdf
2. Uses LLM calls to multimodal models to correct figure caption
fragments that get included in figure images
"""

from dotenv import load_dotenv
load_dotenv()

import logging
import os
import subprocess
import re
import dspy
import json


# Create signatures for image analysis
class ImageCaptionExtractor(dspy.Signature):
    image: dspy.Image = dspy.InputField(desc="The image to analyze")
    question: str = dspy.InputField(desc="Question about text in the image")
    answer: str = dspy.OutputField(desc="Extracted text from the image")


class CaptionFragmentChecker(dspy.Signature):
    text: str = dspy.InputField(desc="Text to analyze")
    question: str = dspy.InputField(desc="Question about the text")
    answer: str = dspy.OutputField(desc="Answer about whether text is a caption fragment")


class PDFTextExtractor:
    def __init__(self, lm, markdown_conversions_folder: str = None, format: str = "markdown"):
        self.markdown_conversions_folder = markdown_conversions_folder
        self.format = format
        self.lm = lm

    def extract_pdf(self, input_pdf_path: str) -> str:
        """Extract text from a single PDF file and convert to markdown using LLM."""
        if not input_pdf_path or not os.path.exists(input_pdf_path):
            logging.error(f"Invalid input PDF path: {input_pdf_path}")
            return None
            
        output_folder = os.path.join(os.path.dirname(input_pdf_path), self.markdown_conversions_folder)
        os.makedirs(output_folder, exist_ok=True)

        filename = os.path.basename(input_pdf_path)
        output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.md")

        command = [
            "marker_single",
            input_pdf_path,
            "--output_dir",
            output_folder,
            "--use_llm",
            "--output_format",
            self.format
        ]

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            logging.info(f"Marker extraction completed for {input_pdf_path}")
            logging.debug(f"Marker output: {result.stdout}")
            
            # Read the generated markdown file
            with open(output_file, 'r', encoding='utf-8') as f:
                markdown_text = f.read()
            
            # Correct image segmentation
            corrected_text = self._correct_image_figure_segmentation(markdown_text, lm)
            
            # Write corrected text back to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(corrected_text)
            
            return output_file
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running marker_single: {e}")
            logging.error(f"Marker stderr: {e.stderr}")
            return None
    
    
    def _correct_image_figure_segmentation(self, text: str, lm) -> str:
        """Correct image and figure segmentation by identifying caption text embedded in images."""
        process_log = []
        
        # Add counter for LLM calls
        llm_call_count = 0
        image_count = 0
        
        # Updated pattern to capture the image markdown and any following text until next image/section
        image_pattern = r'(!\[.*?\]\([^)]+\))\s*(?:([^!*]*)|\*(.*?)\*)'
        
        def process_image_and_caption(match):
            nonlocal image_count, llm_call_count
            image_count += 1
            
            image_markdown = match.group(1)
            # Combine and clean up existing text, removing any stray asterisks
            existing_text = (match.group(2) or match.group(3) or "").strip('* \n')
            
            print(f"\n=== Processing Image {image_count} ===")
            print(f"Image Markdown: {image_markdown}")
            print(f"Existing text: {existing_text.strip()}")
            
            image_path_match = re.search(r'\((.*?)\)', image_markdown)
            if not image_path_match:
                print("ERROR: Could not extract image path")
                return match.group(0)
            
            image_path = image_path_match.group(1)
            base_path = "/home/christian/projects/agents/ai_pi/examples/testwcomments/"
            full_image_path = os.path.join(base_path, image_path)
            print(f"Full image path: {full_image_path}")
            
            predictor_image = dspy.Predict(ImageCaptionExtractor)
            predictor_text = dspy.Predict(CaptionFragmentChecker)
            
            try:
                image = dspy.Image.from_file(full_image_path)
                print("\n=== Making LLM call for image analysis ===")
                llm_call_count += 1
                result = predictor_image(
                    image=image,
                    question="What is the figure number and title shown in this image? If none found, respond with 'No figure title found.'"
                )
                print(f"LLM Response for image analysis: {result.answer}")
                
                embedded_text = result.answer
                print(f"LLM found text: {embedded_text}")
                
                if embedded_text and "No figure title found" not in embedded_text:
                    # Clean up embedded text
                    embedded_text = embedded_text.strip('* \n')
                    
                    # If there's existing text, separate it from the caption
                    if existing_text:
                        # Add double newline between caption and following text
                        final_text = f"{image_markdown}\n\n*{embedded_text}*\n\n{existing_text}"
                    else:
                        final_text = f"{image_markdown}\n\n*{embedded_text}*"
                else:
                    # If no embedded text found but we have existing text
                    final_text = f"{image_markdown}\n\n*{existing_text}*" if existing_text else image_markdown
                
                # Ensure we don't have double asterisks
                final_text = final_text.replace('**', '*')
                
                print(f"Final text:\n{final_text}\n")
                return final_text
                
            except Exception as e:
                print(f"ERROR processing image: {str(e)}")
                return match.group(0)
        
        # Process all images in the text
        corrected_text = re.sub(image_pattern, process_image_and_caption, text)
        
        # Add pattern to fix adjacent captions without proper spacing
        # This will match any caption ending immediately followed by another caption starting
        corrected_text = re.sub(r'(\*[^\n]+?\*)\*([^\n]+?\*)', r'\1\n\n*\2', corrected_text)
        
        print(f"\nTotal images processed: {image_count}")
        print(f"Total LLM calls made: {llm_call_count}")
        
        return corrected_text

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    logging.basicConfig(level=logging.INFO)
    
    # Configure OpenRouter LLM
    openrouter_model = 'openrouter/openai/gpt-4o-mini'
    lm = dspy.LM(
        openrouter_model,
        api_base="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.01,
    )
    dspy.settings.configure(lm=lm)
    filename = "testwocomments.pdf"
    pdf_path = f"/home/christian/projects/agents/ai_pi/examples/{filename}"
    markdown_conversions_folder = f"/home/christian/projects/agents/ai_pi/examples/testwocomments"
    
    extractor = PDFTextExtractor(
        lm=lm,
        markdown_conversions_folder=markdown_conversions_folder,
        format="markdown"
    )
    output_path = extractor.extract_pdf(pdf_path)
    
    print(output_path)

    # # Test _correct_image_figure_segmentation
    # test_markdown_path = "/home/christian/projects/agents/ai_pi/examples/testwcomments/testwcomments.md"
    # output_path = test_markdown_path.replace('.md', '_corrected.md')
    
    # extractor = PDFTextExtractor(lm=lm)
    
    # # Read the test markdown file
    # with open(test_markdown_path, 'r', encoding='utf-8') as f:
    #     test_markdown = f.read()
    
    # # Process the markdown and correct image segmentation
    # corrected_markdown = extractor._correct_image_figure_segmentation(test_markdown, lm)
    
    # # Write corrected markdown to new file
    # with open(output_path, 'w', encoding='utf-8') as f:
    #     f.write(corrected_markdown)
    
    # print(f"Corrected markdown written to: {output_path}")
    
    
