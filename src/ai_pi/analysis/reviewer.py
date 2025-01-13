"""
This file contains the code for an agent which creates suggestions for comments 
and revisions, simulating the style of feedback a PI would give when reviewing a paper.

Requirements:
1. Review that leverages an understanding of the whole paper to provide location-aware feedback
2. Modular assignment of metrics so that the desired qualities can be assessed and modified
    --> see review_criteria.json
"""
import dspy
import json
import logging
from ..lm_config import get_lm_for_task
from dataclasses import dataclass
import os
from ai_pi.utils.logging import setup_logging

@dataclass
class ScoringMetrics:
    """Standardized scoring metrics for paper evaluation"""
    clarity: float
    methodology: float
    novelty: float
    impact: float
    presentation: float
    literature_integration: float
    
    def to_dict(self) -> dict:
        return {k: float(v) for k, v in vars(self).items()}

class ReviewItem(dspy.Signature):
    """Individual review item structure"""
    match_string = dspy.OutputField(desc="Exact text to match")
    comment = dspy.OutputField(desc="Review comment")
    revision = dspy.OutputField(desc="Complete revised text")
    section_type = dspy.OutputField(desc="Section where this review item originated")

class GenerateReviewItemsSignature(dspy.Signature):
    """Signature for generating specific review items and suggested revisions."""
    section_text = dspy.InputField(desc="The text content being reviewed")
    section_type = dspy.InputField(desc="The type of section")
    context = dspy.InputField(desc="Additional context about the paper")
    review_items = dspy.OutputField(
        desc="""List of review items in JSON format. Each item must contain:
        - match_string: The exact text from the paper that needs revision
        - comment: The review comment explaining what should be changed
        - revision: The suggested revised text
        - match_string: the section in which the item was found
        Example format (MUST BE ADHERED TO! PLEASE I BEG OF YOU!):
        [
            {
                "match_string": "exact text from paper",
                "comment": "This sentence needs clarification",
                "revision": "suggested revised text",
                "section_type": "Introduction (ex.)"
            },
            ...
        ]""",
        format=ReviewItem
    )

class SectionReviewerSignature(dspy.Signature):
    """Generate high-level scientific review feedback
    
    Evaluate and review each section based on the text itself,
    the section type, the context surrounding the section and the
    relevant metrics associated with it's evaluation.  
    """
    section_text = dspy.InputField(desc="Text for a particular paper section")
    section_type = dspy.InputField(desc="the type of section, i.e. Methods")
    context = dspy.InputField(desc="Prior context either from a web search summary or from this paper itself, or even from internal LLM knowledge")
    metrics = dspy.OutputField(desc="Output metrics used to evaluate the section", format=ScoringMetrics)
    section_analysis = dspy.OutputField()
    cross_references = dspy.OutputField()
    knowledge_gaps = dspy.OutputField()


class FullDocumentReviewSignature(dspy.Signature):
    """Generate high-level scientific review feedback for a scientific paper.
    
    Apply careful stepwise thinking to each piece of this task:
    
    You are a senior academic reviewer with expertise in providing strategic,
    analytic manuscript feedback.
    
    Tips:
    - Look at the methods, are they the best way to address the hypothesis or research question?
    - Look beyond only what is on the page - consider the role of this research in its field and speak to it's utility in this respect.
    - Think about your scope as the entire research field not just this paper - the key is to make this paper contribute to the field the best it can.
    
    Consider the section type and its expected scope and audience (researchers in this field, so technical jargon is OK). Each 
    section has specific requirements and constraints. Those will be given in specific scopes.       
    """    
    document_text = dspy.InputField(desc="Full document text")
    context = dspy.InputField(desc="Document-wide context and metrics")
    criteria = dspy.InputField(desc="Scoring criteria dictionary")
    metrics = dspy.OutputField(desc="Detailed metrics scores for top-level paper categories", format=ScoringMetrics)
    overall_assessment = dspy.OutputField(desc="Complete paper assessment")
    key_strengths = dspy.OutputField(desc="Major strengths of the paper")
    key_weaknesses = dspy.OutputField(desc="Major weaknesses of the paper")
    global_suggestions = dspy.OutputField(desc="High-level improvement suggestions")

class Reviewer(dspy.Module):
    """carry out overall and section-specific reviews, as well as create specific review items"""
    def __init__(self, verbose=False, validate_reviews=True):
        super().__init__()
        self.logger = logging.getLogger('reviewer')
        self.verbose = verbose
        self.validate_reviews = validate_reviews
        
        # Load review criteria
        try:
            with open("config/review_criteria.json", "r") as f:
                criteria_data = json.load(f)
                self.scoring_criteria = criteria_data.get('scoring_criteria', {})
                self.section_criteria = criteria_data.get('section_criteria', {})
                self.review_quality_criteria = criteria_data.get('review_quality_criteria', {})
        except Exception as e:
            self.logger.error(f"Error loading review criteria: {e}")
            self.scoring_criteria = {}
            self.section_criteria = {}
            self.review_quality_criteria = {}
        
        # Initialize LMs for different review tasks
        self.document_review_lm = get_lm_for_task("document_review")
        self.section_review_lm = get_lm_for_task("section_review")
        
        # Initialize predictors
        self.document_reviewer = dspy.ChainOfThought(FullDocumentReviewSignature)
        self.section_reviewer = dspy.ChainOfThought(SectionReviewerSignature)
        self.review_items_generator = dspy.ChainOfThought(GenerateReviewItemsSignature)


    def review_document(self, document_json: dict, topic_context: str) -> dict:
        try:
            self.logger.info("Starting document review...")
            
            self.topic_context = topic_context
            
            # 1. Build paper-wide understanding first
            self.logger.info("Building paper knowledge...")
            self.paper_knowledge = self._build_paper_knowledge(document_json)
            
            # 2. Generate main review
            self.logger.info("Generating main review...")
            main_review = self._generate_main_review()
            
            # 3. Review individual sections with full context
            self.logger.info("Reviewing individual sections...")
            section_reviews = []
            all_review_items = []  # Create consolidated list
            for section in [s for s in document_json.get('sections', []) if 'references' not in s.get('section_type', '').lower()]:
                section_type = section.get('section_type', '')
                review = self._review_section({
                    'text': section.get('text', ''),
                    'section_type': section_type
                }, paper_context=self.paper_knowledge)
                
                # Extract and parse review items
                if 'review_items' in review:
                    items = review.pop('review_items')
                    try:
                        # If items is a JSON string, parse it
                        if isinstance(items, str):
                            if items.startswith('```json'):
                                # Extract JSON content between backticks
                                json_content = items.split('```json\n')[1].split('\n```')[0]
                            else:
                                json_content = items
                            parsed_items = json.loads(json_content)
                        else:
                            # If it's already a list, use it directly
                            parsed_items = items

                        # Add section information to each review item
                        for item in parsed_items:
                            if isinstance(item, dict):
                                item['section_type'] = section_type
                                all_review_items.append(item)
                            
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Failed to parse review items JSON: {e}")
                        continue
                    except Exception as e:
                        self.logger.error(f"Error processing review items: {e}")
                        continue

                section_reviews.append({
                    'section_type': section_type,
                    'review': review
                })
            
            # Get overall metrics as dictionary
            metrics = self.paper_knowledge.get('metrics', ScoringMetrics(0,0,0,0,0,0))
            if isinstance(metrics, ScoringMetrics):
                metrics_dict = metrics.to_dict()
            else:
                metrics_dict = metrics

            # Structure the output with both overall and section-specific metrics
            document_json['reviews'] = {
                'main_review': main_review,
                'section_reviews': section_reviews,  # Section reviews without review items
                'metrics': metrics_dict
            }
            
            # Store all review items at top level with section information
            document_json['review_items'] = all_review_items
            
            return document_json
            
        except Exception as e:
            self.logger.error(f"Error reviewing document: {str(e)}", exc_info=True)
            raise


    def _build_paper_knowledge(self, document_json: dict) -> dict:
        """Builds comprehensive paper understanding using existing summaries"""
        try:
            hierarchical_summary = document_json['hierarchical_summary']
            
            components = {
                'research_problem': hierarchical_summary['document_summary']['document_analysis'],
                'cross_references': hierarchical_summary['relationship_summary']['relationship_analysis'],
                'sections': document_json['sections'],
                'section_summaries': hierarchical_summary['section_summaries'],
                'full_text': '\n'.join(section['text'] for section in document_json['sections']),
                'topic_context': self.topic_context
            }
            
            # Use context manager for LM
            with dspy.context(lm=self.document_review_lm):
                review_result = self.document_reviewer(
                    document_text=components['full_text'],
                    context={
                        'research_problem': components['research_problem'],
                        'sections': components['sections'],
                        'topic_context': self.topic_context
                    },
                    criteria=self.scoring_criteria
                )
            
            components['metrics'] = review_result.metrics
            
            for summary in hierarchical_summary['section_summaries']:
                if summary['section_type'] == 'Methods':
                    components['key_methods'] = summary['summary']
                elif summary['section_type'] == 'Results':
                    components['main_findings'] = summary['summary']
                elif summary['section_type'] == 'Discussion':
                    components['limitations'] = summary['summary']
            
            return components
                
        except Exception as e:
            self.logger.error(f"Error building paper knowledge: {str(e)}")
            return {}


    def _review_section(self, section: dict, paper_context: dict = None) -> dict:
        """Reviews section using separate predictors for high-level review and specific items"""
        section_context = self._build_section_context(section, paper_context)
        
        with dspy.context(lm=self.section_review_lm):
            result = self.section_reviewer(
                section_text=section['text'],
                section_type=section['section_type'],
                context=section_context
            )
            
            # Add specific instructions for review items
            review_items_result = self.review_items_generator(
                section_text=section['text'],
                section_type=section['section_type'],
                context="""Please identify specific text that needs revision and provide concrete suggestions. 
                          Focus on clarity, accuracy, and scientific rigor. 
                          Return only review items in the specified JSON format."""
            )
            
            # Parse review items
            try:
                if isinstance(review_items_result.review_items, str):
                    json_str = review_items_result.review_items
                    if '```json' in json_str:
                        json_str = json_str.split('```json\n')[1].split('\n```')[0]
                    review_items = json.loads(json_str)
                    
                    # Validate format
                    if not isinstance(review_items, list):
                        self.logger.error("Review items must be a list")
                        review_items = []
                    else:
                        # Filter out invalid items
                        review_items = [
                            item for item in review_items
                            if isinstance(item, dict) and
                            all(key in item for key in ['match_string', 'comment', 'revision'])
                        ]
                else:
                    review_items = []
                    
            except Exception as e:
                self.logger.error(f"Error parsing review items: {e}")
                review_items = []

            self.logger.debug(f"Generated {len(review_items)} valid review items")

            return {
                'metrics': result.metrics,
                'section_analysis': result.section_analysis,
                'review_items': review_items,
                'cross_references': result.cross_references,
                'knowledge_gaps': result.knowledge_gaps
            }
            
    def _build_section_context(self, section: dict, paper_context: dict = None) -> dict:
        """Builds comprehensive context dictionary for section review"""
        if not paper_context:
            return {}
        
        return {
            # Current section info
            'section_type': section['section_type'],
            'section_text': section['text'],
            
            # Direct context from paper knowledge
            'research_problem': paper_context.get('research_problem', ''),
            'main_findings': paper_context.get('main_findings', ''),
            'limitations': paper_context.get('limitations', ''),
            'key_methods': paper_context.get('key_methods', ''),
            
            # Relationship context
            'cross_references': paper_context.get('cross_references', {}),
            
            # Section's place in document
            'section_summary': next(
                (s['summary'] for s in paper_context.get('section_summaries', [])
                 if s['section_type'] == section['section_type']),
                ''
            ),
            
            # Adjacent sections context
            'previous_section': self._get_adjacent_section(paper_context, section['section_type'], -1),
            'next_section': self._get_adjacent_section(paper_context, section['section_type'], 1),
            
            # Document-level metrics for comparison
            'document_metrics': paper_context.get('metrics', {}),
            
            # Section-specific requirements
            'section_requirements': self.section_criteria.get(section['section_type'], {})
        }

    def _get_adjacent_section(self, paper_context: dict, current_section: str, offset: int) -> dict:
        """Helper to get previous/next section context"""
        sections = paper_context.get('sections', [])
        try:
            current_idx = next(i for i, s in enumerate(sections) 
                             if s['section_type'] == current_section)
            adjacent_idx = current_idx + offset
            if 0 <= adjacent_idx < len(sections):
                return {
                    'type': sections[adjacent_idx]['section_type'],
                    'summary': next(
                        (s['summary'] for s in paper_context.get('section_summaries', [])
                         if s['section_type'] == sections[adjacent_idx]['section_type']),
                        ''
                    )
                }
        except StopIteration:
            pass
        return {}


    def _generate_main_review(self) -> dict:
        """Generates the main review using DSPy predictors"""
        with dspy.context(lm=self.document_review_lm):
            result = self.document_reviewer(
                document_text=self.paper_knowledge['full_text'],
                context={
                    'research_problem': self.paper_knowledge['research_problem'],
                    'main_findings': self.paper_knowledge['main_findings'],
                    'sections': self.paper_knowledge['sections']
                },
                criteria=self.scoring_criteria
            )

            return {
                'metrics': result.metrics.to_dict() if isinstance(result.metrics, ScoringMetrics) else result.metrics,
                'overall_assessment': result.overall_assessment,
                'key_strengths': result.key_strengths,
                'key_weaknesses': result.key_weaknesses,
                'global_suggestions': result.global_suggestions
            }


    def _validate_review_quality(self, review_items: list) -> tuple[bool, list]:
        """Validate review quality against defined criteria."""
        # Initialize scores for each criterion
        quality_scores = {
            criterion: 0 for criterion in self.review_quality_criteria.keys()
        }
        
        # Score each review item
        for item in review_items:
            comment = item.comment.lower() if item.comment else ''
            for criterion, terms in self.review_quality_criteria.items():
                quality_scores[criterion] += sum(1 for term in terms if term in comment)
        
        # Generate suggestions for low-scoring criteria
        improvement_suggestions = [
            f"Improve {criterion.replace('_', ' ')} in the review"
            for criterion, score in quality_scores.items()
            if score < len(review_items)
        ]
        
        # Check if all criteria meet minimum threshold
        meets_criteria = all(
            score >= len(review_items) * 0.5 
            for score in quality_scores.values()
        )
        
        return meets_criteria, improvement_suggestions


if __name__ == "__main__":
    import os
    import json
    from dotenv import load_dotenv
    from datetime import datetime
    from pathlib import Path
    load_dotenv()
    
    # Setup logging using the centralized configuration
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    logger = setup_logging(log_dir, timestamp, "reviewer")
    
    reviewer = Reviewer(verbose=True)
    
    # Load and process sample document
    with open("examples/example.json", "r") as f:
        document = json.load(f)

    with open("/home/christian/projects/agents/ai_pi/processed_documents/ScolioticFEPaper_v7_20250113_032648/The_latest_efforts_in_computational_modeling_of_scoliosis_using_finite_element_modeling_-_both_with_and_without_surgical_inte/storm_gen_article_polished.txt", "r") as f:
        topic_context = f.read()
    
    logger.info("Starting document review...")
    reviewed_document = reviewer.review_document(
        document_json=document,
        topic_context=topic_context
    )
    
    # Write results to file
    os.makedirs("ref", exist_ok=True)
    with open("ref/sample_output2.json", "w") as f:
        json.dump(reviewed_document, f, indent=4)
    logger.info("Review complete. Results written to ref/sample_output2.json")
