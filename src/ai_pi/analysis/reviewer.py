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
from typing import List, Dict, Optional
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReviewItem(dspy.Signature):
    """Individual review item structure"""
    match_text = dspy.OutputField(desc="Exact text to match")
    comment = dspy.OutputField(desc="Review comment")
    revision = dspy.OutputField(desc="Complete revised text")

class ReviewerSignature(dspy.Signature):
    """Generate high-level scientific review feedback for a section of text."""
    section_text = dspy.InputField(desc="The text content of the section to review")
    section_type = dspy.InputField(desc="The type of section (e.g., Methods, Results)")
    context = dspy.InputField(desc="Additional context about the paper")
    initial_analysis = dspy.OutputField(desc="Initial high-level analysis of scientific concerns")
    reflection = dspy.OutputField(desc="Self-reflection on whether feedback addresses core scientific issues")
    review_items = dspy.OutputField(desc="List of review items in JSON format", format=list[ReviewItem])
    cross_references = dspy.OutputField(desc="Connections to other sections and related work")
    knowledge_gaps = dspy.OutputField(desc="Identification of remaining knowledge gaps")


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

class ReviewMetrics(dspy.Module):
    """Module for scoring paper sections based on defined criteria"""
    def forward(self, content, criteria):
        """Evaluate content against each metric using defined factors and weights"""
        # Initialize scores
        clarity = self._evaluate_metric(content, criteria['clarity'])
        methodology = self._evaluate_metric(content, criteria['methodology'])
        impact = self._evaluate_metric(content, criteria['impact'])
        presentation = self._evaluate_metric(content, criteria['presentation'])
        literature = self._evaluate_metric(content, criteria['literature_integration'])
        
        # Assess novelty based on impact criteria
        novelty = impact * 0.8  # Slightly reduce impact score for novelty
        
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
            # Use LM to evaluate each factor on a scale of 1-5
            factor_prompt = f"Evaluate the following text on {factor} on a scale of 1-5:\n{content}"
            factor_score = float(self.lm(factor_prompt))  # This will use the configured LM
            total_score += factor_score * weight
            
        return min(5.0, max(0.0, total_score))  # Ensure score is between 0-5

class Reviewer(dspy.Module):
    """Reviews individual sections with awareness of full paper context"""
    
    def __init__(self, 
                 lm: LMConfig = None,
                 verbose: bool = False):
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
            
            # Get metrics as dictionary
            metrics = self.paper_knowledge.get('metrics', ScoringMetrics(0,0,0,0,0,0))
            if isinstance(metrics, ScoringMetrics):
                metrics_dict = metrics.to_dict()
            else:
                metrics_dict = metrics

            # 4. Compile final document with single metrics output
            document_json['reviews'] = {
                'main_review': main_review,
                'section_reviews': section_reviews,
                'metrics': metrics_dict
            }
            
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


    def _review_section(self, section: dict, paper_context: dict = None, custom_criteria: dict = None) -> dict:
        """Reviews section using DSPy predictors with paper-wide context"""
        with dspy.context(lm=self.lm):
            # Include relevant paper context in the review
            section_context = {
                'main_findings': paper_context.get('main_findings', ''),
                'limitations': paper_context.get('limitations', ''),
                'section_type': section['section_type']
            }
            
            # Use section-specific criteria if provided
            section_criteria = custom_criteria or {
                k: v for k, v in self.scoring_criteria.items()
                if k in ['clarity', 'methodology']
            }
            
            # Calculate metrics with context
            metrics = self.metrics_scorer(
                content=section['text'],  # Direct access since we know it exists
                context=section_context,
                criteria=section_criteria
            ).metrics
            
            # Generate analysis with context
            analysis = self.component_operator(
                content=section['text'],  # Direct access since we know it exists
                context=section_context,
                operation_type='analyze',
                target=f"Review the {section['section_type']} section"
            ).result
            
            return {
                'metrics': metrics,
                'review': analysis
            }


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
            return review_items
        
        criteria = self.section_criteria[section_type]
        filtered_items = []
        
        for item in review_items:
            comment = item.get('comment', '').lower()
            revision = item.get('revision', '').lower()
            
            # Check if comment contains any terms to avoid
            should_avoid = any(avoid_term in comment or avoid_term in revision 
                             for avoid_term in criteria['avoid'])
            
            # Check if comment contains focus terms
            has_focus = any(focus_term in comment or focus_term in revision 
                           for focus_term in criteria['focus'])
            
            if has_focus and not should_avoid:
                filtered_items.append(item)
            else:
                logger.debug(f"Filtered out review item for {section_type}: {item}")
        
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
            comment = item.get('comment', '').lower()
            
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
    
    # Print results
    print(json.dumps(reviewed_document, indent=4))
