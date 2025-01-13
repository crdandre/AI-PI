"""
This file uses
https://github.com/stanford-oval/storm

To create a concise-ish summary of the topic at hand to
provide to the review as context
"""

import os
from knowledge_storm import STORMWikiRunnerArguments, STORMWikiRunner, STORMWikiLMConfigs
from knowledge_storm.rm import SerperRM
from ..lm_config import get_lm_for_task
import logging

class StormContextGenerator:
    def __init__(self, 
                 output_dir: str,
                 max_conv_turn: int = 3,
                 max_perspective: int = 3,
                 search_top_k: int = 3,
                 max_thread_num: int = 3,
                 do_research: bool = True,
                 do_generate_outline: bool = True,
                 do_generate_article: bool = True,
                 do_polish_article: bool = True):
        self.output_dir = output_dir
        self.max_conv_turn = max_conv_turn
        self.max_perspective = max_perspective
        self.search_top_k = search_top_k
        self.max_thread_num = max_thread_num
        self.do_research = do_research
        self.do_generate_outline = do_generate_outline
        self.do_generate_article = do_generate_article
        self.do_polish_article = do_polish_article
        
        self._setup_lm_configs()
        self._setup_engine_args()
        self._setup_search_engine()
        
    def _setup_lm_configs(self):
        self.lm_configs = STORMWikiLMConfigs()
        
        question_lm = get_lm_for_task("storm_questions")
        writer_lm = get_lm_for_task("storm_writer")
        
        self.lm_configs.set_conv_simulator_lm(question_lm)
        self.lm_configs.set_question_asker_lm(question_lm)
        self.lm_configs.set_outline_gen_lm(writer_lm)
        self.lm_configs.set_article_gen_lm(writer_lm)
        self.lm_configs.set_article_polish_lm(writer_lm)
        
    def _setup_engine_args(self):
        self.engine_args = STORMWikiRunnerArguments(
            output_dir=self.output_dir,
            max_conv_turn=self.max_conv_turn,
            max_perspective=self.max_perspective,
            search_top_k=self.search_top_k,
            max_thread_num=self.max_thread_num,
        )
        
    def _setup_search_engine(self):
        data = {"autocorrect": True, "num": 10, "page": 1}
        self.rm = SerperRM(
            serper_search_api_key=os.getenv("SERPER_API_KEY"), 
            query_params=data
        )
        
    def generate_context(self, topic: str) -> None:
        logger = logging.getLogger(__name__)
        logger.info(f"Starting STORM context generation for topic: {topic}")
        
        try:
            runner = STORMWikiRunner(self.engine_args, self.lm_configs, self.rm)
            
            logger.info("Running STORM analysis...")
            runner.run(
                topic=topic,
                do_research=self.do_research,
                do_generate_outline=self.do_generate_outline,
                do_generate_article=self.do_generate_article,
                do_polish_article=self.do_polish_article,
            )
            
            logger.info("Running post-processing steps...")
            # runner.post_run()
            
            summary = ""#runner.summary()
            logger.info("STORM context generation complete")
            return self.output_dir, summary
            
        except Exception as e:
            logger.error(f"Error during STORM context generation: {str(e)}")
            raise RuntimeError(f"STORM context generation failed: {str(e)}")
    
    

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    output_dir = "/home/christian/projects/agents/ai_pi/processed_documents/ScolioticFEPaper_v7_20250113_032648"
    
    # Initialize the generator
    generator = StormContextGenerator(
        output_dir=str(output_dir),
        max_conv_turn=2,  # Reduced for faster testing
        max_perspective=2,
        search_top_k=3,
        max_thread_num=2
    )
    
    # Test topic
    test_topic = "The latest efforts in computational modeling of scoliosis using finite element modeling - both with and without surgical interventions."
    
    print(f"\nGenerating STORM analysis for: {test_topic}")
    print(f"Results will be saved to: {output_dir}")
    
    # Generate context and get results
    output_dir, summary = generator.generate_context(test_topic)
    
    print("\nAnalysis complete!")
    print("-" * 50)
    print("\nSummary:")
    print(summary)
    
    
