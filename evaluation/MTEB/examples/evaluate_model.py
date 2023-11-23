import os
import sys
import logging
import argparse
from mteb import MTEB
from InstructorEmbedding import INSTRUCTOR


# AmazonCounterfactualClassification,AmazonPolarityClassification,
# AmazonReviewsClassification,ArguAna,ArxivClusteringP2P,ArxivClusteringS2S,AskUbuntuDupQuestions,
# BIOSSES,BUCC,Banking77Classification,BiorxivClusteringP2P,BiorxivClusteringS2S,CQADupstackAndroidRetrieval,
# CQADupstackEnglishRetrieval,CQADupstackGamingRetrieval,CQADupstackGisRetrieval,CQADupstackMathematicaRetrieval,
# CQADupstackPhysicsRetrieval,CQADupstackProgrammersRetrieval,CQADupstackStatsRetrieval,CQADupstackTexRetrieval,CQADupstackUnixRetrieval,
# CQADupstackWebmastersRetrieval,CQADupstackWordpressRetrieval,ClimateFEVER,DBPedia,EmotionClassification,
# FEVER,FiQA2018,HotpotQA,ImdbClassification,MSMARCO,MSMARCOv2,MTOPDomainClassification,MTOPIntentClassification,
# MassiveIntentClassification,MassiveScenarioClassification,MedrxivClusteringP2P,MedrxivClusteringS2S,MindSmallReranking,
# NFCorpus,NQ,QuoraRetrieval,RedditClustering,RedditClusteringP2P,SCIDOCS,SICK-R,STS12,STS13,STS14,STS15,STS16,STS17,STS22,STSBenchmark,
# SciDocsRR,SciFact,SprintDuplicateQuestions,StackExchangeClustering,StackExchangeClusteringP2P,StackOverflowDupQuestions,SummEval,TRECCOVID,
# Tatoeba,Touche2020,ToxicConversationsClassification,
# TweetSentimentExtractionClassification,TwentyNewsgroupsClustering,TwitterSemEval2015,TwitterURLCorpus.


_tasks = [
    'ArguAna',
    'SciFact',
    'FiQA2018',
    'NFCorpus',
    'QuoraRetrieval'
    # 'MSMARCO',
    # 'HotpotQA',
    #  'DBPedia',
    # 'FEVER'
    # 'NQ',
]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default=None,type=str)
    parser.add_argument('--output_dir', default=None,type=str)
    parser.add_argument('--task_name', default=None,type=str)
    parser.add_argument('--cache_dir', default=None,type=str)
    parser.add_argument('--result_file', default=None,type=str)
    parser.add_argument('--prompt', default=None,type=str)
    parser.add_argument('--split', default='test',type=str)
    parser.add_argument('--batch_size', default=128,type=int)
    args = parser.parse_args()

    if not args.result_file.endswith('.txt') and not os.path.isdir(args.result_file):
        os.makedirs(args.result_file,exist_ok=True)

    # from tqdm import tqdm
    # from functools import partialmethod
    #
    # tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    model = INSTRUCTOR(args.model_name,cache_folder=args.cache_dir)
    evaluation = MTEB(tasks=_tasks, task_langs=["en"])
    evaluation.run(model, output_folder=args.output_dir, eval_splits=[args.split],args=args,)

    print("--DONE--")
