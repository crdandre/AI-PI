"""
This file contains the code for an agent which creates suggestions for comments 
and revisions, simulating the style of feedback a PI would give when reviewing a paper.

Requirements:
1. Review that leverages an understanding of the whole paper to provide location-aware feedback
2. Modular assignment of metrics so that the desired qualities can be assessed and modified
    --> see review_criteria.json
3. 
"""
import dspy
import json
import logging
from ..lm_config import get_lm_for_task, LMConfig
from dataclasses import dataclass
from typing import List
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        return {k: float(v) for k, v in self.__dict__.items()}

@dataclass
class SectionAnalysis:
    """Detailed analysis of a paper section"""
    section_type: str
    content_summary: str
    role_in_paper: str
    key_points: List[str]
    dependencies: List[str]
    supports: List[str]
    metrics: ScoringMetrics

class ReviewItem(dspy.Signature):
    """Individual review item structure"""
    match_string = dspy.OutputField(desc="Exact text to match")
    comment = dspy.OutputField(desc="Review comment")
    revision = dspy.OutputField(desc="Complete revised text")

class ReviewerSignature(dspy.Signature):
    """Generate high-level scientific review feedback for a section of a scientific paper.
    
    Apply careful stepwise thinking to each piece of this task:
    
    You are a senior academic reviewer with expertise in providing strategic,
    analytic manuscript feedback.
    
    First, consider the section type and its expected scope and audience (researchers in this field, so technical jargon is OK). Each section has specific requirements and constraints. Those will be given in specific scopes.
    
    For review items, here's an example of the desired output, to be outputted in JSON format (able to be parsed as json):
    
    review_items: [
        {
            "match_string": "The model predicted curve progression with an average error of 5 degrees.",
            "comment": "The results require statistical validation (e.g., confidence intervals) and 
                discussion of clinical significance. How does this error rate impact treatment decisions?",
            "revision": "The model predicted curve progression with an average error of 5° (95% CI: 3.2-6.8°). 
                This accuracy level is clinically significant as it falls within the threshold needed for 
                reliable treatment planning (< 7°), based on established clinical guidelines."
        }
    ]
        
    """
    section_text = dspy.InputField(desc="The text content of the section being reviewed")
    section_type = dspy.InputField(desc="The type of section (e.g., Abstract, Introduction, Methods, etc.)")
    context = dspy.InputField(desc="Additional context about the paper")
    
    metrics = dspy.OutputField(desc="JSON metrics assessing the section's clarity and methodology")
    initial_analysis = dspy.OutputField(desc="Initial analysis of the section's content")
    reflection = dspy.OutputField(desc="Reflection on the feedback provided")
    review_items = dspy.OutputField(
        desc="List of review items in JSON format",
        format=list[ReviewItem]
    )
    cross_references = dspy.OutputField(desc="Connections to other sections and related work")
    knowledge_gaps = dspy.OutputField(desc="Identification of remaining knowledge gaps")


class ComponentOperationSignature(dspy.Signature):
    """Generic signature for component operations (extract/analyze/generate)"""
    content = dspy.InputField(desc="Content to process")
    operation_type = dspy.InputField(desc="Type of operation (extract/analyze/generate)")
    target = dspy.InputField(desc="Target component or analysis type")
    result = dspy.OutputField(desc="Operation result")

class ReviewMetricsSignature(dspy.Signature):
    """Signature for scoring paper sections based on defined criteria"""
    content = dspy.InputField(desc="Section content to evaluate")
    criteria = dspy.InputField(desc="Scoring criteria dictionary")
    metrics = dspy.OutputField(desc="Detailed metrics scores", format=ScoringMetrics)

class MetricEvaluationSignature(dspy.Signature):
    """Signature for evaluating a single metric factor"""
    content = dspy.InputField(desc="Content to evaluate")
    factor = dspy.InputField(desc="Factor being evaluated")
    score = dspy.OutputField(desc="Numerical score between 1-5", format=float)

class ReviewMetrics(dspy.Module):
    """Module for scoring paper sections based on defined criteria"""
    def __init__(self):
        super().__init__()
        self.evaluator = dspy.Predict(MetricEvaluationSignature)

    def forward(self, content, criteria):
        """Evaluate content against each metric using defined factors and weights"""
        # Initialize scores
        clarity = self._evaluate_metric(content, criteria['clarity'])
        methodology = self._evaluate_metric(content, criteria['methodology'])
        impact = self._evaluate_metric(content, criteria['impact'])
        presentation = self._evaluate_metric(content, criteria['presentation'])
        literature = self._evaluate_metric(content, criteria['literature_integration'])
        
        #TODO: Can implement this based on literature context from scraped sources
        novelty = impact
        
        return self.metrics(ScoringMetrics(
            clarity=clarity,
            methodology=methodology,
            novelty=novelty,
            impact=impact,
            presentation=presentation,
            literature_integration=literature
        ))

    def _evaluate_metric(self, content: str, metric_criteria: dict) -> float:
        """Evaluate a specific metric using its factors and weights"""
        if not content:
            return 0.0
            
        factors = metric_criteria['factors']
        weights = metric_criteria['weights']
        total_score = 0.0
        
        for factor, weight in zip(factors, weights):
            try:
                result = self.evaluator(content=content, factor=factor)
                factor_score = max(1.0, min(5.0, result.score))  # Clamp between 1-5
                total_score += factor_score * weight
            except Exception as e:
                logger.error(f"Failed to evaluate factor {factor}: {e}")
                factor_score = 3.0  # Default to middle score on error
                total_score += factor_score * weight
            
        return min(5.0, max(0.0, total_score))

class Reviewer(dspy.Module):
    """Reviews individual sections with awareness of full paper context"""
    
    def __init__(self, 
                 lm: LMConfig = None,
                 verbose: bool = False,
                 validate_reviews: bool = True):
        super().__init__()
        
        # Get LM for review task using existing configuration
        self.lm = get_lm_for_task("review", lm)
        
        # Configure dspy to use this LM
        dspy.configure(lm=self.lm)
        
        self.paper_knowledge = None
        
        # Load review criteria from JSON
        current_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(current_dir, 'review_criteria.json'), 'r') as f:
            criteria = json.load(f)
            self.section_criteria = criteria['section_criteria']
            self.review_quality_criteria = criteria['review_quality_criteria']
            self.scoring_criteria = criteria['scoring_criteria']
        
        # Configure predictors
        self.reviewer = dspy.Predict(ReviewerSignature)
        self.component_operator = dspy.Predict(ComponentOperationSignature)
        self.metrics_scorer = dspy.Predict(ReviewMetricsSignature)
        
        # Add class type logging
        if verbose:
            print(f"Reviewer type: {type(self.reviewer).__name__}")

        self.verbose = verbose
        self.validate_reviews = validate_reviews


    def review_document(self, document_json: dict) -> dict:
        try:
            logger.info("Starting document review...")
            
            # 1. Build paper-wide understanding first
            logger.info("Building paper knowledge...")
            self.paper_knowledge = self._build_paper_knowledge(document_json)
            
            # 2. Generate main review
            logger.info("Generating main review...")
            main_review = self._generate_main_review()
            
            # 3. Review individual sections with full context
            logger.info("Reviewing individual sections...")
            section_reviews = []
            for section in document_json.get('sections', []):
                logger.info(f"Reviewing section: {section['section_type']}")
                review = self._review_section({
                    'text': section.get('text', ''),
                    'section_type': section.get('section_type', '')
                }, paper_context=self.paper_knowledge)
                section_reviews.append({
                    'section_type': section['section_type'],
                    'review': review
                })
            
            # Get overall metrics as dictionary
            metrics = self.paper_knowledge.get('metrics', ScoringMetrics(0,0,0,0,0,0))
            if isinstance(metrics, ScoringMetrics):
                metrics_dict = metrics.to_dict()
            else:
                metrics_dict = metrics

            # Flatten review items for easier consumption
            all_review_items = []
            for section_review in section_reviews:
                if isinstance(section_review['review']['review_items'], list):
                    all_review_items.extend(section_review['review']['review_items'])

            # Structure the output with both overall and section-specific metrics
            document_json['reviews'] = {
                'main_review': main_review,
                'section_reviews': section_reviews,  # Each section review contains its own metrics
                'metrics': metrics_dict  # Overall document metrics
            }
            
            # Store the flattened review items in a separate key for convenience
            document_json['review_items'] = all_review_items
            
            return document_json
            
        except Exception as e:
            logger.error(f"Error reviewing document: {str(e)}", exc_info=True)
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
                'full_text': '\n'.join(section['text'] for section in document_json['sections'])
            }
            
            # Overall document metrics
            metrics_result = self.metrics_scorer(
                content=components['full_text'],
                criteria=self.scoring_criteria
            )
            
            components['metrics'] = metrics_result.metrics
            
            for summary in hierarchical_summary['section_summaries']:
                if summary['section_type'] == 'Methods':
                    components['key_methods'] = summary['summary']
                elif summary['section_type'] == 'Results':
                    components['main_findings'] = summary['summary']
                elif summary['section_type'] == 'Discussion':
                    components['limitations'] = summary['summary']
            
            return components
                
        except Exception as e:
            logger.error(f"Error building paper knowledge: {str(e)}")
            return {}


    def _review_section(self, section: dict, paper_context: dict = None) -> dict:
        """Reviews section using DSPy predictors with comprehensive paper-wide context"""
        
        # Build comprehensive section context
        section_context = {
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
            'document_metrics': paper_context.get('metrics', {})
        }

        with dspy.context(lm=self.lm):
            # Enhanced reviewer prompt with richer context
            result = self.reviewer(
                section_text=section['text'],
                section_type=section['section_type'],
                context=section_context
            )
            
            # Calculate section-specific metrics with full context
            metrics = self.metrics_scorer(
                content=section['text'],
                context=section_context,
                criteria=self._get_section_specific_criteria(section['section_type'])
            ).metrics

            return {
                'metrics': metrics,
                'initial_analysis': result.initial_analysis,
                'reflection': result.reflection,
                'review_items': self._validate_section_specific_feedback(
                    result.review_items,
                    section['section_type']
                ),
                'cross_references': result.cross_references,
                'knowledge_gaps': result.knowledge_gaps
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

    def _get_section_specific_criteria(self, section_type: str) -> dict:
        """Get section-specific review criteria"""
        base_criteria = self.scoring_criteria.copy()
        if section_type in self.section_criteria:
            base_criteria.update(self.section_criteria[section_type])
        return base_criteria


    def _generate_main_review(self) -> dict:
        """Generates the main review using DSPy predictors"""
        with dspy.context(lm=self.lm):
            metrics = self.paper_knowledge['metrics']
            if isinstance(metrics, ScoringMetrics):
                metrics = metrics.to_dict()

            content = {
                'text': self.paper_knowledge['full_text'],
                'sections': self.paper_knowledge['sections'],
                'research_problem': self.paper_knowledge['research_problem'],
                'main_findings': self.paper_knowledge['main_findings'],
                'metrics': metrics
            }

            return {
                'overall_assessment': self.component_operator(
                    content=content,
                    operation_type='generate',
                    target='overall_assessment'
                ).result,
                'key_strengths': self.component_operator(
                    content=content,
                    operation_type='generate',
                    target='strengths'
                ).result,
                'key_weaknesses': self.component_operator(
                    content=content,
                    operation_type='generate',
                    target='weaknesses'
                ).result,
                'global_suggestions': self.component_operator(
                    content=content,
                    operation_type='generate',
                    target='suggestions'
                ).result
            }


    def _validate_section_specific_feedback(self, review_items: list, section_type: str) -> list:
        """Filter review items based on section-specific criteria."""
        if section_type not in self.section_criteria:
            logger.warning(f"Unknown section type: {section_type}")
            return []
        
        criteria = self.section_criteria[section_type]
        filtered_items = []
        
        for item in review_items:
            if not isinstance(item, ReviewItem):
                continue
            
            comment = item.comment.lower() if item.comment else ''
            revision = item.revision.lower() if item.revision else ''
            
            # Check if comment contains any terms to avoid
            should_avoid = any(avoid_term in comment or avoid_term in revision 
                             for avoid_term in criteria['avoid'])
            
            # Check if comment contains focus terms
            has_focus = any(focus_term in comment or focus_term in revision 
                           for focus_term in criteria['focus'])
            
            if has_focus and not should_avoid:
                filtered_items.append(item)
        
        return filtered_items


    def _validate_review_quality(self, review_items: list) -> tuple[bool, list]:
        """Validate review quality against defined criteria."""
        quality_scores = {
            'depth': 0,
            'integration': 0,
            'specificity': 0
        }
        
        improvement_suggestions = []
        
        # Analyze each review item against quality criteria
        for item in review_items:
            comment = item.comment.lower() if item.comment else ''
            
            # Check depth
            depth_score = sum(1 for term in self.review_quality_criteria['depth'] 
                            if term in comment)
            quality_scores['depth'] += depth_score
            
            # Check integration
            integration_score = sum(1 for term in self.review_quality_criteria['integration'] 
                                 if term in comment)
            quality_scores['integration'] += integration_score
            
            # Check specificity
            specificity_score = sum(1 for term in self.review_quality_criteria['specificity'] 
                                  if term in comment)
            quality_scores['specificity'] += specificity_score
        
        # Generate improvement suggestions if scores are low
        if quality_scores['depth'] < len(review_items):
            improvement_suggestions.append("Deepen theoretical analysis and methodological critique")
        if quality_scores['integration'] < len(review_items):
            improvement_suggestions.append("Strengthen connections across sections and to broader literature")
        if quality_scores['specificity'] < len(review_items):
            improvement_suggestions.append("Provide more concrete examples and actionable suggestions")
        
        # Return True if all quality criteria meet minimum thresholds
        meets_criteria = all(score >= len(review_items) * 0.5 
                           for score in quality_scores.values())
        
        return meets_criteria, improvement_suggestions


if __name__ == "__main__":
    import os
    import json
    from dotenv import load_dotenv
    load_dotenv()
    
    reviewer = Reviewer(verbose=True)
    
    # Load and process sample document
    with open("processed_documents/ScolioticFEPaper_v7_20250107_011921/ScolioticFEPaper_v7_reviewed.json", "r") as f:
        document = json.load(f)
    
    # Review entire document
    reviewed_document = reviewer.review_document(document)
    
    # Write results to file
    os.makedirs("ref", exist_ok=True)
    with open("ref/sample_output2.json", "w") as f:
        json.dump(reviewed_document, f, indent=4)
