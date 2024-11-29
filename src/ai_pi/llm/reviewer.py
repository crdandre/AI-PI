"""
This file contains the code for an agent which, given relevant context
from prior reviews and any other guidelines for the review process, can
create suggestions for comments and revisions, simulating the style of feedback
a PI would given when reviewing a paper

For now, some manual token management for fitting within context length...
"""
import os
import dspy
import json
import re

class ReviewerAgent:
    def __init__(
        self,
        llm=dspy.LM(
            "nvidia_nim/meta/llama3-70b-instruct",
            api_key=os.environ['NVIDIA_API_KEY'],
            api_base=os.environ['NVIDIA_API_BASE']
        ),
    ):
        self.llm = llm
        dspy.configure(lm=llm)
        self.call_review = dspy.ChainOfThought("question -> answer")
        self.call_revise = dspy.ChainOfThought(
            "question -> revision_list: list[list[str]]"
        )


class SectionReviewer:
    """Reviews individual sections with awareness of full paper context"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def review_section(
        self,
        section_text: str,
        section_type: str,
        paper_context: dict
    ) -> dict:
        """
        Review a section using the paper's overall context.
        Returns structured review data ready for document output.
        """
        # First, let's add debug logging
        print(f"\nReviewing {section_type} section...")
        
        prompt = f"""Review this {section_type} section of an academic paper and provide specific suggestions for improvement.
        
        Paper Context: {paper_context.get('paper_summary', '')}
        
        Section Text: {section_text}
        
        Provide your review as a JSON object with the following structure:
        {{
            "review_items": [
                {{
                    "match_text": "exact text to match",
                    "comment": "your review comment",
                    "revision": "suggested revision (provide complete revised text, not just changes)"
                }}
            ]
        }}
        
        Important:
        - For each revision, provide the complete revised text, not just the changes
        - Ensure match_text exists exactly in the source text
        - Keep suggestions focused and specific
        """
        
        # Get the response from the LLM
        response = self.llm.complete(prompt)
        
        # Convert the CompletionResponse to a string
        response_text = str(response)
        print(f"\nRaw LLM Response:\n{response_text}\n")
        
        review_items = {
            'match_strings': [],
            'comments': [],
            'revisions': []
        }

        try:
            # Strip any markdown code block markers from the response
            cleaned_response = response_text.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            
            # Parse the cleaned JSON
            parsed_response = json.loads(cleaned_response)
            print(f"Successfully parsed JSON response")
            
            for item in parsed_response.get('review_items', []):
                match_text = item.get('match_text', '').replace('"', '"').replace('"', '"')
                match_text = ' '.join(match_text.split())
                
                if match_text and match_text in section_text:  # Validate match exists
                    review_items['match_strings'].append(match_text)
                    review_items['comments'].append(item.get('comment', ''))
                    review_items['revisions'].append(item.get('revision', ''))
                    print(f"Added review item for text: {match_text[:50]}...")
            
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {str(e)}")
            print(f"Failed to parse response: {response_text}")
            # Try to recover by looking for common patterns
            try:
                # Simple recovery attempt if the response isn't proper JSON
                if '"text":' in response_text:
                    matches = re.findall(r'"text":\s*"([^"]+)"', response_text)
                    comments = re.findall(r'"comment":\s*"([^"]+)"', response_text)
                    revisions = re.findall(r'"revision":\s*"([^"]+)"', response_text)
                    
                    for i, match in enumerate(matches):
                        review_items['match_strings'].append(match)
                        if i < len(comments):
                            review_items['comments'].append(comments[i])
                        if i < len(revisions):
                            review_items['revisions'].append(revisions[i])
                    
                    print("Recovered review items using regex")
                    print(json.dumps(review_items, indent=2))
            except Exception as regex_error:
                print(f"Recovery attempt failed: {str(regex_error)}")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            
        return {'review': review_items}


if __name__ == "__main__":
    # Test the reviewer
    from llama_index.llms.nvidia import NVIDIA
    
    llm = NVIDIA(model="meta/llama3-70b-instruct")
    reviewer = SectionReviewer(llm)
    
    test_context = {
        'paper_summary': 'Test paper about scoliosis modeling.'
    }
    
    test_section = {
        'text': 'This is a test section.',
        'type': 'Methods'
    }
    
    review = reviewer.review_section(
        test_section['text'],
        test_section['type'],
        test_context
    )
    print(review)