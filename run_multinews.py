from summarizer import SummPip
import argparse
from utils import read_file 
import os

def read_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="minh_vlsp/Data/")
    parser.add_argument("--source_file", type=str, default="clean_concat_test_data.csv")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--w2v_file", type=str, default="minh_vlsp/SummPipForVie/embedded_word/word2vec_vi_words_300dims.txt")
    parser.add_argument("--lm_file", type=str, default="/home/yenvt/minhnt/SenBert/vn_sbert_deploy/phobert_base_mean_tokens_NLI_STS")
    # parser.add_argument("--cluster", type=str, default="spc")
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--output_file", type=str, default="SumpipVie/testSummary.txt")
    parser.add_argument("--nb_words", type=int, default=15)
    parser.add_argument("--nb_clusters", type=int, default=9,help="for spectral clustering")
    parser.add_argument("--ita",type=float,default=0.83)
    return parser.parse_args()

def main():
    
    args=read_arguments() 
    nb_words = args.nb_words
    nb_clusters = args.nb_clusters
    seed = args.seed
    path = args.input_path
    # clus_alg = args.cluster
    w2v_file = args.w2v_file
    lm_file = args.lm_file
    ita = args.ita

    src_file = args.source_file
    src_list = read_file(path, src_file)
    print("Number of instances: ", len(src_list))

    pipe = SummPip(nb_clusters=nb_clusters, nb_words=nb_words, ita=ita, seed=seed, w2v_file=w2v_file, lm_path=lm_file, use_lm=True)
    summary_list = pipe.summarize(src_list)

    out_path = args.output_path
    outfile = args.output_file
    f = open(os.path.join(out_path, outfile), "w")
    print("summary list length",len(summary_list))
    summary_list = [line.replace("\n","") +"\n" for line in summary_list]
    f.writelines(summary_list)
    f.close()

if __name__ == "__main__":
    main()