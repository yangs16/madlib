#include "postgres.h"
#include "funcapi.h"
#include "fmgr.h"
#include "catalog/pg_type.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "executor/executor.h"
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#ifndef NO_PG_MODULE_MAGIC
PG_MODULE_MAGIC;
#endif
/* Indicate "version 1" calling conventions for all exported functions. */
PG_FUNCTION_INFO_V1(pldaPlusRandomAssign);
PG_FUNCTION_INFO_V1(pldaPlusCountTopic);
PG_FUNCTION_INFO_V1(pldaPlusInt32ArrayAdd);
PG_FUNCTION_INFO_V1(pldaPlusArrayFoldAdd);
PG_FUNCTION_INFO_V1(pldaPlusGibbsSFunc);
PG_FUNCTION_INFO_V1(pldaPlusGibbsFFunc);
PG_FUNCTION_INFO_V1(pldaPlusGibbsPred);

/************************************************************************ 
 * Begin: Experiment with window function to transfer states between rows 
 * Using encoded int32 array should be more efficient 
*************************************************************************/ 
Datum pldaPlusGibbsFFunc(PG_FUNCTION_ARGS);
Datum pldaPlusGibbsFFunc(PG_FUNCTION_ARGS)
{
	ArrayType * arr_state = PG_GETARG_ARRAYTYPE_P(0);
	int32 * state = (int32 *)ARR_DATA_PTR(arr_state);

	int32 offset = state[0];
	int32 payload_len = state[offset];

	ArrayType * arr_result = construct_array(NULL, payload_len, INT4OID, 4, true, 'i');
	int32 * result = (int32 *)ARR_DATA_PTR(arr_result);
	memcpy(result, state + offset + 1, payload_len * sizeof(int32));

	PG_RETURN_ARRAYTYPE_P(arr_result);
}

typedef struct {
	int32 * offset;
	int32 * corpus_topic;
	int32 * word_topic;
	int32 * payload_len;
	int32 * doc_topic_topics;
} __plda_plus_gibbs_state;

static int32 __pldaPlusSampleTopic(int32, int32, int32 *, int32 *, int32 *, float8, float8); 

/**
 * state int4[]
 * word_count int4, words int4[], counts int4[], doc_topic_topics int4[],
 * word_topic int4[], corpus_topic int4[], 
 * alpha float, beta float,
 * voc_size int4, topic_num int4)
**/
Datum pldaPlusGibbsSFunc(PG_FUNCTION_ARGS);
Datum pldaPlusGibbsSFunc(PG_FUNCTION_ARGS)
{
	const int32 __default_doc_len = 1000;
	__plda_plus_gibbs_state ppgs;

	int32 word_count = PG_GETARG_INT32(1);
	ArrayType * arr_words = PG_GETARG_ARRAYTYPE_P(2);
	ArrayType * arr_counts = PG_GETARG_ARRAYTYPE_P(3);
	ArrayType * arr_doc_topic_topics = PG_GETARG_ARRAYTYPE_P(4);

 	int32 * words = (int32 *)ARR_DATA_PTR(arr_words);
 	int32 * counts = (int32 *)ARR_DATA_PTR(arr_counts);
 	int32 * doc_topic_topics = (int32 *)ARR_DATA_PTR(arr_doc_topic_topics);

	ArrayType * arr_word_topic;
	int32 * word_topic;
	if(!PG_ARGISNULL(5)){
		arr_word_topic = PG_GETARG_ARRAYTYPE_P(5);
	 	word_topic = (int32 *)ARR_DATA_PTR(arr_word_topic);
	}

	ArrayType * arr_corpus_topic;
 	int32 * corpus_topic;
	if(!PG_ARGISNULL(6)){
		arr_corpus_topic = PG_GETARG_ARRAYTYPE_P(6);
 		corpus_topic = (int32 *)ARR_DATA_PTR(arr_corpus_topic);
	}	

	float8 alpha = PG_GETARG_FLOAT8(7);
	float8 beta = PG_GETARG_FLOAT8(8);
	int32 voc_size = PG_GETARG_INT32(9);
	int32 topic_num = PG_GETARG_INT32(10);
	
	int32 state_array_fixed_len = 2 + (voc_size + 2) * topic_num;
	int32 corpus_topic_offset = 1;
	int32 word_topic_offset = 1 + topic_num;
	int32 payload_len_offset = 1 + (voc_size + 1) * topic_num;
	int32 doc_topic_topics_offset = 2 + (voc_size + 1) * topic_num; 

	ArrayType * arr_return_state;	
	if(PG_ARGISNULL(0)){
		int32 prealloc_size = state_array_fixed_len + (int32)(ceil(((float)word_count)/__default_doc_len) * __default_doc_len);
		arr_return_state = construct_array(NULL, prealloc_size, INT4OID, 4, true, 'i');
		int32 * return_state = (int32 *) ARR_DATA_PTR(arr_return_state);

		ppgs.offset = return_state;
		ppgs.corpus_topic = return_state + corpus_topic_offset;
		ppgs.word_topic = return_state + word_topic_offset;
		ppgs.payload_len = return_state + payload_len_offset;
		ppgs.doc_topic_topics = return_state + doc_topic_topics_offset;

		memcpy(ppgs.corpus_topic, corpus_topic, topic_num * sizeof(int32));
		memcpy(ppgs.word_topic, word_topic, voc_size * topic_num * sizeof(int32));
	}else{
		ArrayType *  arr_curr_state = PG_GETARG_ARRAYTYPE_P(0);
		int32 * curr_state = (int32 *)ARR_DATA_PTR(arr_curr_state);
		int32 curr_preallo_size = ARR_DIMS(arr_curr_state)[0];

		if(word_count > curr_preallo_size - state_array_fixed_len){
			int32 prealloc_size = state_array_fixed_len + (int32)(ceil(((float)word_count)/__default_doc_len) * __default_doc_len);
			arr_return_state = construct_array(NULL, prealloc_size, INT4OID, 4, true, 'i');
			int32 * return_state = (int32 *) ARR_DATA_PTR(arr_return_state);

			ppgs.offset = return_state;
			ppgs.corpus_topic = return_state + corpus_topic_offset;
			ppgs.word_topic = return_state + word_topic_offset;
			ppgs.payload_len = return_state + payload_len_offset;
			ppgs.doc_topic_topics = return_state + doc_topic_topics_offset;

			memcpy(ppgs.corpus_topic, curr_state + corpus_topic_offset, topic_num * sizeof(int32));
			memcpy(ppgs.word_topic, curr_state + word_topic_offset, voc_size * topic_num * sizeof(int32));
		}else{
			arr_return_state = arr_curr_state;
			ppgs.offset = curr_state;
			ppgs.corpus_topic = curr_state + corpus_topic_offset;
			ppgs.word_topic = curr_state + word_topic_offset;
			ppgs.payload_len = curr_state + payload_len_offset;
			ppgs.doc_topic_topics = curr_state + doc_topic_topics_offset;
		}
	}
	*(ppgs.offset) = payload_len_offset;
	*(ppgs.payload_len) = topic_num + word_count;
	memcpy(ppgs.doc_topic_topics, doc_topic_topics, (topic_num + word_count) * sizeof(int32));

	int32 unique_word_count = ARR_DIMS(arr_words)[0];
	int32 word_index = topic_num;
	for(int32 i = 0; i < unique_word_count; i++) {
		int32 wordid = words[i];
		for(int32 j = 0; j < counts[i]; j++){
			int32 topic = ppgs.doc_topic_topics[word_index];
			int32 retopic = __pldaPlusSampleTopic(topic_num, topic, ppgs.doc_topic_topics, ppgs.word_topic + wordid * topic_num, ppgs.corpus_topic, alpha, beta);
			ppgs.doc_topic_topics[word_index] = retopic;

			ppgs.corpus_topic[topic]--;
			ppgs.corpus_topic[retopic]++;
			ppgs.doc_topic_topics[topic]--;
			ppgs.doc_topic_topics[retopic]++;
			ppgs.word_topic[wordid * topic_num + topic]--;
			ppgs.word_topic[wordid * topic_num + retopic]++;
			word_index++;
		}
	}

	PG_RETURN_ARRAYTYPE_P(arr_return_state);
}
/************************************************************************ 
 * End
 ************************************************************************/ 

/**
 * word_count int4, words int4[], counts int4[], doc_topic_topics int4[],
 * word_topic int4[], corpus_topic int4[], 
 * alpha float, beta float,
 * topic_num int4, iter_num int4)
**/
Datum pldaPlusGibbsPred(PG_FUNCTION_ARGS);
Datum pldaPlusGibbsPred(PG_FUNCTION_ARGS)
{
	int32 word_count = PG_GETARG_INT32(0);
	ArrayType * arr_words = PG_GETARG_ARRAYTYPE_P(1);
	ArrayType * arr_counts = PG_GETARG_ARRAYTYPE_P(2);
	ArrayType * arr_doc_topic_topics = PG_GETARG_ARRAYTYPE_P(3);
	ArrayType * arr_word_topic = PG_GETARG_ARRAYTYPE_P(4);
	ArrayType * arr_corpus_topic = PG_GETARG_ARRAYTYPE_P(5);
	float8 alpha = PG_GETARG_FLOAT8(6);
	float8 beta = PG_GETARG_FLOAT8(7);
	int32 topic_num = PG_GETARG_INT32(8);
	int32 iter_num = PG_GETARG_INT32(9);

 	int32 * words = (int32 *)ARR_DATA_PTR(arr_words);
 	int32 * counts = (int32 *)ARR_DATA_PTR(arr_counts);
 	int32 * doc_topic_topics = (int32 *)ARR_DATA_PTR(arr_doc_topic_topics);
	int32 * word_topic = (int32 *)ARR_DATA_PTR(arr_word_topic);
 	int32 *	corpus_topic = (int32 *)ARR_DATA_PTR(arr_corpus_topic);

	ArrayType * arr_result = construct_array(NULL, topic_num + word_count, INT4OID, 4, true, 'i');
	int32 * result = (int32 *) ARR_DATA_PTR(arr_result);
	memcpy(result, doc_topic_topics, (topic_num + word_count) * sizeof(int32));

	for(int it = 0; it < iter_num; it++){
		int32 unique_word_count = ARR_DIMS(arr_words)[0];
		int32 word_index = topic_num;
		for(int32 i = 0; i < unique_word_count; i++) {
			int32 wordid = words[i];
			for(int32 j = 0; j < counts[i]; j++){
				int32 topic = result[word_index];
				int32 retopic = __pldaPlusSampleTopic(topic_num, topic, result, word_topic + wordid * topic_num, corpus_topic, alpha, beta);
				result[word_index] = retopic;
				result[topic]--;
				result[retopic]++;
				word_index++;
			}
		}
	}

	PG_RETURN_ARRAYTYPE_P(arr_result);
}

Datum pldaPlusRandomAssign(PG_FUNCTION_ARGS);
Datum pldaPlusRandomAssign(PG_FUNCTION_ARGS)
{
	int32 word_count = PG_GETARG_INT32(0);
	int32 topic_num = PG_GETARG_INT32(1);

	if(word_count < 1 || topic_num < 1)
		elog(ERROR, "Word count or topic number should be no less than 1.");

	ArrayType * arr_result = construct_array(NULL, topic_num + word_count, INT4OID, 4, true, 'i');
	int32 * result = (int32 *)ARR_DATA_PTR(arr_result);

	for(int32 i = 0; i < word_count; i++){
		int32 topic = rand() % topic_num;
		result[topic] += 1;
		result[topic_num + i] = topic;	
	}

	PG_RETURN_ARRAYTYPE_P(arr_result);
}

/**
 * This function samples a new topic for a given word based on count statistics
 * computed on the rest of the corpus. This is the core function in the Gibbs
 * sampling inference algorithm for LDA. 
 * 
 * Parameters
 *  @param num_topics	number of topics
 *  @param topic	the current assigned topic of the word
 *  @param count_w_z	the word-topic count vector
 *  @param count_d_z	the distribution of topics in the current document
 *  @param count_z	the distribution of number of words in the corpus assigned to each topic
 *  @param alpha	the Dirichlet parameter for the topic multinomial
 *  @param beta		the Dirichlet parameter for the per-topic word multinomial
 *
 * The function is non-destructive to all the input arguments.
 */
static int32 __pldaPlusSampleTopic(int32 num_topics, int32 topic, int32 * count_d_z, int32 * count_w_z, int32 * count_z, float8 alpha, float8 beta) 
{
	// this array captures the cumulative prob. distribution of the topics
	float8 * topic_prs = (float8 *)palloc(sizeof(float8) * num_topics); 

	/* calculate topic (unnormalised) probabilities */
	float8 total_unpr = 0;
	for (int32 i = 0; i < num_topics; i++) {
		int32 nwz = count_w_z[i];
		int32 ndz = count_d_z[i];
		int32 nz = count_z[i];

		// adjust the counts to exclude current word's contribution
		if (i == topic) {
			nwz--;
			ndz--;
			nz--;
		}

		// probability
		float unpr = (ndz + alpha) * (nwz + beta) / (nz + num_topics * beta);
		total_unpr += unpr;
		topic_prs[i] = total_unpr;
	}

	/* normalise probabilities */
	for (int32 i = 0; i < num_topics; i++)
		topic_prs[i] /= total_unpr;

	/* Draw a topic at random */
	float8 r = drand48();
	int32 ret = 1;
	while (true) {
		if (ret == num_topics || r < topic_prs[ret-1]) break;
		ret++; 
	}
	if (ret < 1 || ret > num_topics)
		elog(ERROR, "sampleTopic: ret = %d", ret);

	pfree(topic_prs);
	return ret - 1;
}


Datum pldaPlusCountTopic(PG_FUNCTION_ARGS);
Datum pldaPlusCountTopic(PG_FUNCTION_ARGS)
{
	if (!(fcinfo->context && IsA(fcinfo->context, AggState)))
		elog(ERROR, "pldaPlusCountTopic not used as part of an aggregate");

	int32 voc_size = PG_GETARG_INT32(4);
	int32 topic_num = PG_GETARG_INT32(5);
	ArrayType * arr_state;
	if (PG_ARGISNULL(0)) {
		arr_state = construct_array(NULL, voc_size * topic_num, INT4OID, 4, true, 'i');
	} else {
		arr_state = PG_GETARG_ARRAYTYPE_P(0);
	}
	int32 * state = (int32 *)ARR_DATA_PTR(arr_state);

	ArrayType * arr_words = PG_GETARG_ARRAYTYPE_P(1);
	ArrayType * arr_counts = PG_GETARG_ARRAYTYPE_P(2);
	ArrayType * arr_topics = PG_GETARG_ARRAYTYPE_P(3);

	int32 unique_word_count = ARR_DIMS(arr_words)[0];
	int32 * words = (int32 *)ARR_DATA_PTR(arr_words);
	int32 * counts = (int32 *)ARR_DATA_PTR(arr_counts);
	int32 * topics = (int32 *)ARR_DATA_PTR(arr_topics);

	int32 word_index = topic_num;
	for(int32 i = 0; i < unique_word_count; i++){
		int32 wordid = words[i];
		for(int32 j = 0; j < counts[i]; j++){
			int32 topic = topics[word_index];
			state[wordid * topic_num + topic]++;
			word_index++;
		}
	}

	PG_RETURN_ARRAYTYPE_P(arr_state);
}

Datum pldaPlusInt32ArrayAdd(PG_FUNCTION_ARGS);
Datum pldaPlusInt32ArrayAdd(PG_FUNCTION_ARGS)
{
	ArrayType * arr_state1 = PG_GETARG_ARRAYTYPE_P(0);
	ArrayType * arr_state2 = PG_GETARG_ARRAYTYPE_P(1);

	int32 * state1 = (int32 *)ARR_DATA_PTR(arr_state1);
	int32 * state2 = (int32 *)ARR_DATA_PTR(arr_state2);
	int32 count = ARR_DIMS(arr_state1)[0];
	for(int32 i = 0; i < count; i++) {
		state1[i] += state2[i];
	}

	PG_RETURN_ARRAYTYPE_P(arr_state1);
}

Datum pldaPlusArrayFoldAdd(PG_FUNCTION_ARGS);
Datum pldaPlusArrayFoldAdd(PG_FUNCTION_ARGS)
{
	ArrayType * arr_array = PG_GETARG_ARRAYTYPE_P(0);
	int32 * array = (int32 *)ARR_DATA_PTR(arr_array);
	int32 len = ARR_DIMS(arr_array)[0];
	int32 size = PG_GETARG_INT32(1);

	if(size < 0)
		elog(ERROR, "the size should be larger than 0");
	if(size >= len)
		PG_RETURN_ARRAYTYPE_P(arr_array);

	ArrayType * arr_result = construct_array(NULL, size, INT4OID, 4, true, 'i');
	int32 * result = (int32 *)ARR_DATA_PTR(arr_result);
	for(int32 i = 0; i < len; i++) {
		result[i % size] += array[i];
	}

	PG_RETURN_ARRAYTYPE_P(arr_result);
}
