"""
This file uses
https://github.com/stanford-oval/storm

To create a concise-ish summary of the topic at hand to
provide to the review as context
"""

import os
import logging
from knowledge_storm import STORMWikiRunnerArguments, STORMWikiRunner, STORMWikiLMConfigs
from knowledge_storm.rm import SerperRM
from big_dict_energy.lm_setup import LMForTask
from big_dict_energy.utils.logging import setup_logging

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
        self.logger = logging.getLogger('storm_context')
        self.logger.info("Initializing StormContextGenerator")
        
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
        self.logger.debug("Setting up LM configurations")
        self.lm_configs = STORMWikiLMConfigs()
        
        # Create LM instances for different tasks
        question_lm = LMForTask.STORM_QUESTIONS.get_lm()
        writer_lm = LMForTask.STORM_WRITER.get_lm()
        
        # Set the LMs for different STORM components
        self.lm_configs.set_conv_simulator_lm(question_lm)
        self.lm_configs.set_question_asker_lm(question_lm)
        self.lm_configs.set_outline_gen_lm(writer_lm)
        self.lm_configs.set_article_gen_lm(writer_lm)
        self.lm_configs.set_article_polish_lm(writer_lm)
        
    def _setup_engine_args(self):
        self.logger.debug("Setting up STORM engine arguments")
        self.engine_args = STORMWikiRunnerArguments(
            output_dir=self.output_dir,
            max_conv_turn=self.max_conv_turn,
            max_perspective=self.max_perspective,
            search_top_k=self.search_top_k,
            max_thread_num=self.max_thread_num,
        )
        
    def _setup_search_engine(self):
        self.logger.debug("Setting up search engine")
        data = {"autocorrect": True, "num": 10, "page": 1}
        self.rm = SerperRM(
            serper_search_api_key=os.getenv("SERPER_API_KEY"), 
            query_params=data
        )
        
    def generate_context(self, topic: str) -> None:
        self.logger.info(f"Starting STORM context generation for topic: {topic}")
        
        try:
            runner = STORMWikiRunner(self.engine_args, self.lm_configs, self.rm)
            
            self.logger.info("Running STORM analysis...")
            runner.run(
                topic=topic,
                do_research=self.do_research,
                do_generate_outline=self.do_generate_outline,
                do_generate_article=self.do_generate_article,
                do_polish_article=self.do_polish_article,
            )
            
            self.logger.info("Running post-processing steps...")
            # runner.post_run()
            
            summary = ""#runner.summary()
            self.logger.info("STORM context generation complete")
            return self.output_dir, summary
            
        except Exception as e:
            self.logger.error(f"Error during STORM context generation: {str(e)}")
            self.logger.exception("Full traceback:")
            raise RuntimeError(f"STORM context generation failed: {str(e)}")

if __name__ == "__main__":
    from pathlib import Path
    from datetime import datetime
    
    # Setup logging using the centralized configuration
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    logger = setup_logging(log_dir, timestamp, "storm_context")
    
    # Example usage
    output_dir = "examples/storm_output"
    generator = StormContextGenerator(output_dir=output_dir)
    
    try:
        output_dir, summary = generator.generate_context("Finite Element Analysis in Biomechanics")
        logger.info(f"Context generated successfully. Output directory: {output_dir}")
    except Exception as e:
        logger.error(f"Failed to generate context: {str(e)}")
    
    
