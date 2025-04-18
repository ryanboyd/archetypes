# -*- coding: utf-8 -*-
import csv
import torch
import os.path
import numpy as np
from math import log
from tqdm import tqdm
from statistics import variance
from torch import mean as torchmean
from sentence_transformers import SentenceTransformer, util
from nltk import sent_tokenize
from pprint import pprint

def weighted_mean(input: torch.tensor, weight: list=[]) -> torch.tensor:
    '''
    Calculate the weighted mean of a stack of tensors
    :param input: a stack of tensors to average
    :param weight: a list of weights
    :return: a tensor of the weighted mean
    '''
    # Default weights to 1.0 if not provided
    if not weight or len(input) != len(weight):
        weight = [1.0] * len(input)
    input_tensor = input.clone().detach().to('cpu')
    weight_tensor = torch.tensor(weight).to('cpu')
    w_sum = torch.sum(input_tensor * weight_tensor.unsqueeze(-1), dim=0)
    w_mean = w_sum / sum(weight)
    return w_mean
        

def make_safe_filename(s):
    def safe_char(c):
        if c.isalnum():
            return c
        else:
            return "_"
    return "".join(safe_char(c) for c in s).rstrip("_")

def z_center(input_vector: list) -> list:
    list_mean = sum(input_vector) / len(input_vector)
    list_stdev = sum([((x - list_mean) ** 2) for x in input_vector]) / len(input_vector) ** 0.5
    z_list = [(x - list_mean / list_stdev) for x in input_vector]
    return z_list

def mean_center(input_vector: list) -> list:
    list_mean = sum(input_vector) / len(input_vector)
    centered_list = [x - list_mean for x in input_vector]
    return centered_list

def cronbach(input_vectors: list) -> float:
    k = len(input_vectors)
    if k < 2:
        return None
    item_variances = []
    for i in range(0, k):
        item_variances.append(variance(input_vectors[i]))
    total_variance = variance(np.sum(input_vectors, 0))
    cronbach_alpha = (k / (k - 1)) * ((total_variance - sum(item_variances)) / total_variance)
    return cronbach_alpha


# define a class to hold our archetypes for each construct.
class ArchetypeCollection():

    def __init__(self, ) -> None:
        """
        The collection itself is empty upon initialization
        """
        self.archetype_names = []
        self.archetype_sentences = {}
        self.archetype_weights = {}
        return

    def add_archetype(self,
                      name: str,
                      sentences: list,
                      weights: list = []):
        """

        :param name: The name of the archetype being added
        :param sentences: A list of sentences that are used to represent the archetype
        :return:
        """
        # wipe out the existing entries if we already have an archetype by this name
        if name in self.archetype_names:
            self.archetype_names.remove(name)
            self.archetype_sentences.pop(name)
            self.archetype_weights.pop(name)

        self.archetype_names.append(name)
        self.archetype_sentences[name] = sentences
        self.archetype_weights[name] = weights

        print(f"Archetype added: {name}")

    def add_archetypes_from_CSV(self,
                                filepath: str,
                                file_encoding:str ="utf-8-sig",
                                file_has_headers: bool=True):

        archetype_sentence_dict = {}
        archetype_weights_dict = {}
        archetype_list = []

        with open(filepath, 'r', encoding=file_encoding) as fin:
            csvr = csv.reader(fin)
            if file_has_headers:
                _ = csvr.__next__()

            for row in csvr:

                archetype_name = row[0].strip()
                prototype_sentence = row[1].strip()
                if len(row) > 2:
                    archetype_weight = float(row[2])
                else:
                    archetype_weight = 1.0
                    
                # New archetype
                if archetype_name not in archetype_sentence_dict.keys(): 
                    archetype_sentence_dict[archetype_name] = []
                    archetype_weights_dict[archetype_name] = []
                    archetype_list.append(archetype_name)

                # Store the prototypical sentences and weights
                archetype_sentence_dict[archetype_name].append(prototype_sentence)
                archetype_weights_dict[archetype_name].append(archetype_weight)

        for archetype_name in archetype_list:
            self.add_archetype(name=archetype_name, sentences=archetype_sentence_dict[archetype_name], weights=archetype_weights_dict[archetype_name])


class ArchetypeResult():

    def __init__(self,
                 sentence_text: str,
                 sentence_embedding,
                 WC: int) -> None:
        
        self.WC = WC
        self.sentence_text = sentence_text
        self.sentence_embedding = sentence_embedding
        self.error_encountered = False
        self.archetype_scores = {}

# this is the main machine that will do the actual scoring of the texts
class ArchetypeQuantifier():

    def __init__(self,
                 archetypes: ArchetypeCollection,
                 model: str,
                 hf_token: str = None) -> None:
        """
        Initialize an instance of the ArchetypeQuantifier class
        :param archetypes: An instance of the Archetype_Collection class
        :param model: The name of the sentence-transformers model that you want to use to quantify archetypes
        :param hf_token: Your huggingface token, if necessary.
        """
        self.results = []
        self.archetypes = archetypes
        self.model = SentenceTransformer(model, use_auth_token=hf_token)

        # take the archetype sentences and convert each one to an embedding.
        # then, we calculate the average embedding for each archetype construct.
        self.archetype_embeddings = {}
        self.archetype_order = {}

        order_count = 0
        for archetype_name in self.archetypes.archetype_names:
            input = self.model.encode(
                sentences=self.archetypes.archetype_sentences[archetype_name],
                convert_to_tensor=True
            )
            weight = self.archetypes.archetype_weights.get(archetype_name,[1.0]*len(self.archetypes.archetype_sentences[archetype_name]))
            #self.archetype_embeddings[archetype_name] = torchmean(input=input,axis=0).tolist()
            self.archetype_embeddings[archetype_name] = weighted_mean(input=input, weight=weight).tolist()

            self.archetype_order[order_count] = archetype_name
            order_count += 1

    def evaluate_archetype_consistency(self,
                                       mean_center_vectors: bool = False) -> None:
        """
        Print output that shows something like the "internal consistency" of each archetype, based on the prototypical sentences
        :return:
        """

        for archetype_name in self.archetypes.archetype_names:

            print(f"Evaluating {archetype_name}...")

            mean_cos_sim = 0.0
            num_sentences = len(self.archetypes.archetype_sentences[archetype_name])
            archetype_weights = self.archetypes.archetype_weights.get(archetype_name,[1.0]*num_sentences)
            sum_weights = sum(archetype_weights)
            

            sentence_vectors = []

            for i, archetype_sentence in enumerate(self.archetypes.archetype_sentences[archetype_name]):
                
                archetype_test_sent = [archetype_sentence]
                archetype_weight = archetype_weights[i]
                archetype_rest_sents = [x for x in self.archetypes.archetype_sentences[archetype_name] if
                                        x != archetype_test_sent]

                # calculate the embedding for the 'test' sentence
                archetype_test_embedding = torchmean(self.model.encode(
                                                     archetype_test_sent,
                                                     convert_to_tensor=True),
                                                     axis=0).tolist()

                # we save these vectors for later so that we can calculate cronbach's alpha
                sentence_vectors.append(archetype_test_embedding)

                # calculate the average embedding for all of the 'rest' sentences
                archetype_rest_embedding = torchmean(self.model.encode(
                                                     archetype_rest_sents,
                                                     convert_to_tensor=True),
                                                     axis=0).tolist()

                if mean_center_vectors:
                    archetype_test_embedding = mean_center(archetype_test_embedding)
                    archetype_rest_embedding = mean_center(archetype_rest_embedding)
                    

                cos_sim = float(util.pytorch_cos_sim(archetype_test_embedding,
                                                     archetype_rest_embedding)[0])
                
                weighted_cos_sim = cos_sim * archetype_weight

                mean_cos_sim += weighted_cos_sim / sum_weights

                print(f"\t{round(weighted_cos_sim, 5)}: {archetype_test_sent[0]}")

            print("\t--------------------")
            if mean_center_vectors:
                print(f"\t{round(mean_cos_sim, 5)}: Average item-rest correlation")
            else:
                print(f"\t{round(mean_cos_sim, 5)}: Average item-rest cosine similarity")

            cronbachs_alpha = cronbach(sentence_vectors)
            if cronbachs_alpha is None:
                print("\tCannot calculate Cronbach's alpha where the number of sentences < 2")
            else:
                print(f"\t{round(cronbach(sentence_vectors), 5)}: Cronbach's alpha\n\n")


    def export_all_archetype_vectors(self,
                                     output_file_location: str,
                                     mean_center_vectors: bool = False) -> None:

        """
        This function exports the raw vectors for your archetypes/prototypes in transposed format.
        :param output_file_location: The name of the file where you would like your exported results to be written.
        :param mean_center_vectors: Do you want to mean-center your vectors for these calculations?
        :return:
        """

        raw_vectors = []
        vector_names = []

        # start off by getting the vectors for all of the archetypes
        for archetype_name in self.get_list_of_archetypes():
            archetype_weights = self.archetypes.archetype_weights.get(archetype_name,[])
            vector_names.append(f"Archetype: {archetype_name}")
            raw_vec = weighted_mean(
                input = self.model.encode(
                    self.archetypes.archetype_sentences[archetype_name],
                    convert_to_tensor=True),
                weight=archetype_weights).tolist()
            raw_vectors.append(raw_vec)

        # now we do it for the individual sentences
        for archetype_name in self.get_list_of_archetypes():
            for archetype_sentence in self.archetypes.archetype_sentences[archetype_name]:
                vector_names.append(f"Prototype ({archetype_name}): {archetype_sentence}")
                raw_vec = torchmean(self.model.encode(
                    [archetype_sentence],
                    convert_to_tensor=True),
                    axis=0).tolist()
                raw_vectors.append(raw_vec)

        if mean_center_vectors:
            for i in range(0, len(raw_vectors)):
                raw_vectors[i] = mean_center(raw_vectors[i])

        raw_vectors_trasposed = list(map(list, zip(*raw_vectors)))

        with open(output_file_location, 'w', encoding='utf-8-sig', newline='') as fout:
            csvw = csv.writer(fout)
            csvw.writerow(vector_names)

            for row in raw_vectors_trasposed:
                csvw.writerow(row)

        print("All archetype vectors have been exported.")


    def export_all_archetype_relationships(self,
                                           output_file_location: str,
                                           mean_center_vectors: bool = False) -> None:
        """
        This function exports cosine similarities (or correlations, if you mean center the vectors) between all aspects
        of your archetypes. This is a way to view the relationships between the semantics of not only the archetypes,
        but all of the prototype sentences as well.
        :param output_file_location: The name of the file where you would like your exported results to be written.
        :param mean_center_vectors: Do you want to mean-center your vectors for these calculations?
        :return:
        """

        print("Calculating all relationships within/across all archetypes...")

        raw_vectors = []
        vector_names = []

        # start off by getting the vectors for all of the archetypes
        for archetype_name in self.get_list_of_archetypes():
            archetype_weights = self.archetypes.archetype_weights.get(archetype_name,[])
            vector_names.append(f"Archetype: {archetype_name}")
            raw_vec = weighted_mean(
                input = self.model.encode(
                    self.archetypes.archetype_sentences[archetype_name],
                    convert_to_tensor=True),
                weight=archetype_weights).tolist()
            raw_vectors.append(raw_vec)


        # now we do it for the individual sentences
        for archetype_name in self.get_list_of_archetypes():
            for archetype_sentence in self.archetypes.archetype_sentences[archetype_name]:
                vector_names.append(f"Prototype ({archetype_name}): {archetype_sentence}")
                raw_vec = torchmean(self.model.encode(
                                    [archetype_sentence],
                                    convert_to_tensor=True),
                                    axis=0).tolist()
                raw_vectors.append(raw_vec)

        if mean_center_vectors:
            for i in range(0, len(raw_vectors)):
                raw_vectors[i] = mean_center(raw_vectors[i])

        corr_matrix = np.corrcoef(raw_vectors).tolist()

        with open(output_file_location, 'w', encoding='utf-8-sig', newline='') as fout:
            csvw = csv.writer(fout)

            csv_header_row = [""]
            csv_header_row.extend(vector_names)

            csvw.writerow(csv_header_row)

            for i in range(0, len(corr_matrix)):
                output_row = [vector_names[i]]
                output_row.extend(corr_matrix[i])
                csvw.writerow(output_row)

        print(f"All relationships exported to: {output_file_location}")


    def export_intra_archetype_correlations(self,
                                               output_folder: str,
                                               mean_center_vectors: bool = False) -> None:
        """

        :param output_folder: Choose a folder to export separate CSVs of the correlations within archetypes
        :param mean_center_vectors: Do you want to mean-center your vectors? If yes, results can be interpreted as correlations. If not, these should be interpreted as cosine similarities.
        :return:
        """

        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        for archetype_name in self.get_list_of_archetypes():

            matrix_size = len(self.archetypes.archetype_sentences[archetype_name])

            corr_matrix = [[None] * matrix_size for x in range(matrix_size)]

            for i in range(0, matrix_size):
                corr_matrix[i][i] = 1.0

            archetype_embeddings = self.model.encode(sentences=self.archetypes.archetype_sentences[archetype_name],
                                                     convert_to_tensor=True)

            for i in range(0, matrix_size):
                for j in range(i + 1, matrix_size):

                    archetype_i_embedding = archetype_embeddings[i].tolist()
                    archetype_j_embedding = archetype_embeddings[j].tolist()

                    if mean_center_vectors:
                        archetype_i_embedding = mean_center(archetype_i_embedding)
                        archetype_j_embedding = mean_center(archetype_j_embedding)

                    correl = float(util.pytorch_cos_sim(archetype_i_embedding,
                                                         archetype_j_embedding)[0])

                    corr_matrix[i][j] = correl
                    corr_matrix[j][i] = correl
                    #print(corr_matrix)

            header_row = ['']

            header_row.extend(self.archetypes.archetype_sentences[archetype_name])

            try:
                with open(os.path.join(output_folder, f"{make_safe_filename(archetype_name)}.csv"), 'w', encoding='utf-8-sig', newline='') as fout:
                    csvw = csv.writer(fout)
                    csvw.writerow(header_row)
                    for i in range(0, matrix_size):
                        output_row = [self.archetypes.archetype_sentences[archetype_name][i]]
                        output_row.extend(corr_matrix[i])
                        csvw.writerow(output_row)
            except Exception as e:
                print(e, '\nError! Could not open file to write correlations.')
                return

            print(f"Successfully exported intra-archetype cosine similarity matrix for: {archetype_name}")

        return

    def export_inter_archetype_correlations(self, output_filename: str) -> None:
        #get archetype names
        archetype_names = self.archetypes.archetype_names
        #matrix of cos sim
        cos_sim_matrix = [[None] * len(archetype_names) for x in range(len(archetype_names))]
        #fill in the diagonal with 1s
        for i in range(0, len(archetype_names)):
            cos_sim_matrix[i][i] = 1.0
        #get the embeddings for each archetype
        for archetype_construct, archetype_embedding in self.archetype_embeddings.items():
            #get cos sim between each pair of archetypes
            for other_archetype_construct, other_archetype_embedding in self.archetype_embeddings.items():
                cos_sim = float(util.pytorch_cos_sim(archetype_embedding, other_archetype_embedding)[0])
                cos_sim_matrix[archetype_names.index(archetype_construct)][archetype_names.index(other_archetype_construct)] = cos_sim
        #write to csv
        try:
            with open(output_filename, 'w', encoding='utf-8-sig', newline='') as fout:
                csvw = csv.writer(fout)
                csvw.writerow(archetype_names)
                for i in range(0, len(archetype_names)):
                    output_row = [archetype_names[i]]
                    output_row.extend(cos_sim_matrix[i])
                    csvw.writerow(output_row)
        except Exception as e:
            print(e,'\nError! Could not open file to write correlations.')
            return
        print(f"Successfully exported inter-archetype cosine similarity matrix for archetypes to: {output_filename}")

    def get_list_of_archetypes(self, ) -> list:
        """
        Return a list or the archetype names, in order
        :return:
        """
        archetype_names = []

        for i in range(len(self.archetypes.archetype_names)):
            archetype_names.append(self.archetype_order[i])

    def batch_analyze_to_csv(self,
                             texts: list,
                             text_metadata: dict,
                             csv_sent_output_location: str,
                             csv_doc_output_location: str,
                             append_to_existing_csv: bool = False,
                             output_encoding: str = "utf-8-sig",
                             mean_center_vectors: bool = False,
                             fisher_z_transform: bool = False,
                             doc_avgs_exclude_sents_with_WC_less_than: int = 0,
                             doc_level_aggregation_type: str = "mean"):
        """

        :param texts: a list of texts that you want to analyze
        :param text_metadata: a dictionary where each key is the name of the metadata variable, and the value is a list of metadata items that correspond to the input texts
        :param csv_sent_output_location: path where you want to save a CSV of your sentence-level output
        :param csv_doc_output_location: path where you want to save a CSV of your document-level output
        :param append_to_existing_csv: do you want to append to an existing CSV file?
        :param output_encoding: the file encoding that you want to use to write your CSV files
        :param mean_center_vectors: do you want to mean-center your vectors during the analysis?
        :param fisher_z_transform: Do you want to Fisher Z-transform the cosine similarities to help ensure a more normal distribution of measures?
        :param doc_avgs_exclude_sents_with_WC_less_than: when calculating document-level averages, sentences with fewer than N words will be excluded. Note that these exclusions will only be reflected in the document-level averages for each archetype, but not in other values (e.g., word count)
        :param doc_level_aggregation_type: when aggregating scores at the document level, do you want document-level "mean" or document-level "max"?
        :return:
        """

        writemode = 'w'
        if append_to_existing_csv:
            writemode = 'a'

        with open(csv_sent_output_location, writemode, encoding=output_encoding,
                  newline='') as fout_sent, open(csv_doc_output_location, writemode,
                                                 encoding=output_encoding, newline='') as fout_doc:

            csvw_sent = csv.writer(fout_sent)
            csvw_doc = csv.writer(fout_doc)

            meta_headers = list(text_metadata.keys())

            if append_to_existing_csv is False:
                csvw_sent.writerow(self.generate_csv_header_sentence_level(
                    metadata_headers=meta_headers,))
                csvw_doc.writerow(self.generate_csv_header_document_level(
                    metadata_headers=meta_headers,
                    aggregation_type=doc_level_aggregation_type))

            for i in tqdm(range(len(texts))):

                self.analyze(texts[i],
                             mean_center_vectors=mean_center_vectors,
                             fisher_z_transform=fisher_z_transform)

                meta_output = []

                for meta_item in meta_headers:
                    meta_output.append(text_metadata[meta_item][i])

                csvw_sent.writerows(self.generate_csv_output_sentence_level(
                    input_metadata=meta_output))
                csvw_doc.writerow(self.generate_csv_output_document_level(
                    input_metadata=meta_output,
                    doc_avgs_exclude_sents_with_WC_less_than=doc_avgs_exclude_sents_with_WC_less_than,
                    aggregation_type=doc_level_aggregation_type))

    def analyze(self, text: str,
                mean_center_vectors: bool = False,
                fisher_z_transform: bool = False) -> None:
        """
        Takes the input text, segments into sentences, then analyzes each sentence for similarity to each archetype
        :param text: The text that you want to analyze
        :param mean_center_vectors: Do you want to mean-center your vectors when calculating similarities?
        :param fisher_z_transform: Do you want to Fisher Z-transform the cosine similarities to help ensure a more normal distribution of measures?
        :return:
        """

        # take the input text and tokenize into sentences
        sentences = sent_tokenize(str(text).strip())
        # make sure there are no empty sentences
        sentences = [i for i in sentences if i.strip()]

        # set up a list that will contain our results
        results = []
        error_encountered = False

        sentence_embeddings = None
        try:
            # attempt to convert input sentences to embeddings
            sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True).tolist()
        except Exception as e:
            print(e, "\nError was encountered when trying to embed sentences.")
            error_encountered = True

        # calculate similarity between each sentence and each archetype construct
        for i in range(len(sentences)):

            # set up an ArchetypeResult object to hold our results for this sentence
            archetype_result = ArchetypeResult(sentence_text=sentences[i],
                                               sentence_embedding=None,
                                               WC=len(sentences[i].strip().split()))

            for archetype_construct, archetype_embedding in self.archetype_embeddings.items():

                if error_encountered:
                    # if we encountered an error when trying to create sentence embeddings,
                    # we'll just store the results as empty.
                    archetype_result.archetype_scores[archetype_construct] = None

                else:
                    # otherwise, if everything above went well, we'll calculate the
                    # cosine similarity between the sentence embedding and each archetype embedding...

                    # first, we keep a copy of the sentence embedding
                    archetype_result.sentence_embedding = sentence_embeddings[i]

                    cos_sim = None

                    if mean_center_vectors:
                        archetype_result.sentence_embedding = mean_center(archetype_result.sentence_embedding)

                        cos_sim = float(util.pytorch_cos_sim(mean_center(archetype_embedding),
                                                       archetype_result.sentence_embedding)[0])
                    else:
                        cos_sim = float(util.pytorch_cos_sim(archetype_embedding,
                                                       archetype_result.sentence_embedding)[0])

                    if not fisher_z_transform:
                        archetype_result.archetype_scores[archetype_construct] = cos_sim
                    else:
                        # if we have a value that is equal to (negative) one,
                        # then we just pretend that the correlation is .9999999999999999, which would give us a
                        # Fisher Z value of (-)18.714973875118524
                        if cos_sim >= 1.0:
                            archetype_result.archetype_scores[archetype_construct] = 18.714973875118524
                        elif cos_sim <= -1.0:
                            archetype_result.archetype_scores[archetype_construct] = -18.714973875118524
                        else:
                            archetype_result.archetype_scores[archetype_construct] = .5 * log(
                                (1 + cos_sim) / (1 - cos_sim))

            archetype_result.error_encountered = error_encountered
            results.append(archetype_result)

        self.results = results

    def get_raw_results(self, ) -> list:
        """
        Returns a list of the class ArchetypeResult, where each element in the list corresponds to each sentence in the input text, in order
        :return:
        """
        return self.results

    def get_results_per_sentence(self, ) -> list:
        """
        Returns a list of the scores for each archetype for each sentence.  Each value is in the same order as the archetype names
        :return:
        """
        results = []

        for result in self.results:

            sentence_result = [result.WC]

            if not result.error_encountered:

                for i in range(len(self.archetype_order.keys())):
                        sentence_result.append(result.archetype_scores[self.archetype_order[i]])

                results.append(sentence_result)

    def get_results_text_avgs(self,
                              doc_avgs_exclude_sents_with_WC_less_than: int = 0,
                              aggregation_type: str = "mean") -> list:
        """
        Calculates the average of each archetype across all sentences in the text
        :param doc_avgs_exclude_sents_with_WC_less_than: when calculating document-level averages, sentences with fewer than N words will be excluded. Note that these exclusions will only be reflected in the document-level averages for each archetype, but not in other values (e.g., word count)
        :return:
        """

        sentence_results = self.get_results_per_sentence()


        # retain sentences with a WC greater than our desired threshold
        sentence_results_clean = [sent_result for sent_result in sentence_results if sent_result[0] >= doc_avgs_exclude_sents_with_WC_less_than]


        # only calculate the actual averages if we have more than zero sentences
        if len(sentence_results_clean) > 0:
            sentence_results_as_np_array = np.array(sentence_results_clean)
            if aggregation_type == "max":
                results_avg = np.max(sentence_results_as_np_array, axis=0).tolist()
            else:
                results_avg = np.average(sentence_results_as_np_array, axis=0).tolist()

        else:
            results_avg = [None] * len(self.get_list_of_archetypes())

        # this is getting the sum of the word count, which is why we're just doing the sum here.
        # we include short sentences that may have been omitted from the archetype averages  in this sum because we
        # want an accurate reflection of the overall text length.
        results_avg[0] = 0
        for res in sentence_results:
            results_avg[0] += res[0]

        return results_avg


    def generate_csv_header_sentence_level(self,
                                           metadata_headers: list):
        """
        Helper function to generate a CSV header
        :param metadata_headers: The other headers that will be prepended to your list of archetypes
        :return:
        """
        mh = metadata_headers.copy()
        mh.extend(["text", "WC"])
        mh.extend(self.get_list_of_archetypes())
        return mh

    def generate_csv_header_document_level(self,
                                           metadata_headers: list,
                                           aggregation_type: str = "mean"):

        header_data_cos_sim = self.get_list_of_archetypes()
        if aggregation_type == "max":
            header_data_cos_sim = [x + "_cossim_max" for x in header_data_cos_sim]
        else:
            header_data_cos_sim = [x + "_cossim_avg" for x in header_data_cos_sim]

        mh = metadata_headers.copy()
        mh.extend(["NumSentences", "WC"])
        mh.extend(header_data_cos_sim)

        return mh

    def generate_csv_output_sentence_level(self, input_metadata: list) -> list:

        sentence_level_results = self.get_results_per_sentence()

        output_data = []

        for i in range(len(sentence_level_results)):
            sentence_level_output_data = []
            sentence_level_output_data.extend(input_metadata)
            sentence_level_output_data.append(self.results[i].sentence_text)
            sentence_level_output_data.extend(sentence_level_results[i])
            output_data.append(sentence_level_output_data)

        return output_data

    def generate_csv_output_document_level(self, 
                                           input_metadata: list,
                                           raw_counts: bool = False,
                                           doc_avgs_exclude_sents_with_WC_less_than: int = 0,
                                           aggregation_type: str = "mean") -> list:

        output_data = input_metadata
        output_data.append(str(len(self.results)))

        if aggregation_type == "max":
            output_data.extend(self.get_results_text_avgs(doc_avgs_exclude_sents_with_WC_less_than,
                                                          aggregation_type=aggregation_type))
        else:
            output_data.extend(self.get_results_text_avgs(doc_avgs_exclude_sents_with_WC_less_than,
                               aggregation_type="mean"))


        return output_data

if __name__ == "__main__":
    '''
    Example usage of the ArchetypeCollection and ArchetypeQuantifier classes
    '''

    model_name = 'dmlls/all-mpnet-base-v2-negation'
    archetypes = ArchetypeCollection()

    archetypes.add_archetype(name="Positive Archetype",
                             sentences=["I feel like I have a lot of control over my life."],
                             weights=[1.0])

    archetypes.add_archetype(name="Negative Archetype",
                             sentences=["I don't feel like I have a lot of control over my life."])

    archetype_quantifier = ArchetypeQuantifier(archetypes=archetypes,
                                               model=model_name)

    example_texts = ["Sometimes, I just feel like I don't know if anything that I do has any effect.",
                     "I think I am the master of my own destiny.",
                     "I feel like I am just a leaf in the wind, blowing wherever the breeze takes me."]
    
    for example_text in example_texts:
        
        print(f"\n-> Analyzing text: {example_text}")

        print("Instantiating ArchetypeQuantifier.")
        archetype_quantifier.analyze(example_text,
                                    mean_center_vectors=True,
                                    fisher_z_transform=False)

        results = archetype_quantifier.results

        for result in results:
            print(f"Sentence Text: {result.sentence_text}")
            print(f"Word Count: {result.WC}")
            print("Archetype scores:")
            pprint(result.archetype_scores)
            print("\n")
