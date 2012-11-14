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
PG_FUNCTION_INFO_V1(randomAssign);
PG_FUNCTION_INFO_V1(sampleNewTopic);
PG_FUNCTION_INFO_V1(sumTopicCount);
PG_FUNCTION_INFO_V1(logLikelihood);

static float8 log_beta(float8 alpha, int32 len){
	if (alpha <= 0 || len <= 0)
		elog(ERROR, "Invalid parameter alpha and len for log_beta: %.f\t %d", alpha, len);
		
	return len * lgamma(alpha) - lgamma(len * alpha);
}

static float8 log_multi_beta(float8 * pTopics, int32 len){
	if (len <= 0){
		elog(ERROR, "Invalid parameter len for log_multi_beta: %d", len);
	}
	for(int32 i = 0; i < len; i++){
		if(pTopics[i] <= 0)
			elog(ERROR, "Invalid parameter for log_multi_beta: %d\t%f", i, pTopics[i]);
	}

	float8 res = 0.0;
	float8 sum = 0.0;
	for(int32 i = 0; i < len; i++){
		res += lgamma(pTopics[i]);
		sum += pTopics[i];
	}
	res -= lgamma(sum);
	return res;
}

Datum randomAssign(PG_FUNCTION_ARGS);
Datum randomAssign(PG_FUNCTION_ARGS)
{
	int32 word_cnt = PG_GETARG_INT32(0);
	int32 num_topic = PG_GETARG_INT32(1);

	if(word_cnt < 0 || num_topic < 0)
		elog(ERROR, "Word count or topic number should be no less than 1.");

	Datum * arr1 = palloc0(word_cnt * sizeof(Datum));
	ArrayType *	ret_topics_arr = construct_array(arr1, word_cnt, INT4OID, 4, true, 'i');
	int32 * ret_topics = (int32 *)ARR_DATA_PTR(ret_topics_arr);

	for(int32 i = 0; i < word_cnt; i++)
		ret_topics[i] = rand() % num_topic;

	PG_RETURN_ARRAYTYPE_P(ret_topics_arr);
}

Datum logLikelihood(PG_FUNCTION_ARGS);
Datum logLikelihood(PG_FUNCTION_ARGS)
{
	float8 ll = PG_GETARG_FLOAT8(0);
	float8 hyper = PG_GETARG_FLOAT8(2);

	ArrayType * arr_topics = PG_GETARG_ARRAYTYPE_P(1);
	int32 * topics = (int32 *)ARR_DATA_PTR(arr_topics);
	int32 num_topics = ARR_DIMS(arr_topics)[0];

	float8 * x = (float8 *)palloc(sizeof(float8) * num_topics); 
	for(int32 i = 0; i < num_topics; i ++)
		x[i] = topics[i] + hyper;

	ll += log_multi_beta(x, num_topics);
	ll -= log_beta(hyper, num_topics);

	pfree(x);

	PG_RETURN_FLOAT8(ll);
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
 *  @param eta		the Dirichlet parameter for the per-topic word multinomial
 *
 * The function is non-destructive to all the input arguments.
 */
static int32 sampleTopic(int32 num_topics, int32 topic, int32 * count_d_z, int32 * count_w_z, int32 * count_z, float8 alpha, float8 eta) 
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
		float unpr = (ndz + alpha) * (nwz + eta) / (nz + num_topics * eta);
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


/**
 * This function assigns a topic to each word in a document using the count
 * statistics obtained so far on the corpus. The function returns an array
 * of int4s, which pack two arrays together: the topic assignment to each
 * word in the document (the first len elements in the returned array), and
 * the number of words assigned to each topic (the last num_topics elements
 * of the returned array).
 */
Datum sampleNewTopic(PG_FUNCTION_ARGS);
Datum sampleNewTopic(PG_FUNCTION_ARGS)
{
	ArrayType * arr_topics = PG_GETARG_ARRAYTYPE_P(0);
	ArrayType * arr_count_d_z = PG_GETARG_ARRAYTYPE_P(1);
	ArrayType * arr_count_w_z = PG_GETARG_ARRAYTYPE_P(2);
	ArrayType * arr_count_z = PG_GETARG_ARRAYTYPE_P(3);

	int32 num_topics = ARR_DIMS(arr_count_z)[0];

	int32 * topics = (int32 *)ARR_DATA_PTR(arr_topics);
	int32 * count_d_z = (int32 *)ARR_DATA_PTR(arr_count_d_z);
	int32 * count_w_z = (int32 *)ARR_DATA_PTR(arr_count_w_z);
	int32 * count_z = (int32 *)ARR_DATA_PTR(arr_count_z);

	float8 alpha = PG_GETARG_FLOAT8(4);
	float8 eta = PG_GETARG_FLOAT8(5);

	ArrayType * ret_topics_arr;
	int32 * ret_topics;

	int32 count = ARR_DIMS(arr_topics)[0];
	Datum * arr1 = palloc0(count * sizeof(Datum));
	ret_topics_arr = construct_array(arr1, count, INT4OID, 4, true, 'i');
	ret_topics = (int32 *)ARR_DATA_PTR(ret_topics_arr);

	for(int32 i = 0; i < count; i++) {
		int32 topic = topics[i];
		int32 rtopic = sampleTopic(num_topics, topic, count_d_z, count_w_z, count_z, alpha, eta);
		ret_topics[i] = rtopic;
	}

	PG_RETURN_ARRAYTYPE_P(ret_topics_arr);
}

Datum sumTopicCount(PG_FUNCTION_ARGS);
Datum sumTopicCount(PG_FUNCTION_ARGS)
{
	if (!(fcinfo->context && IsA(fcinfo->context, AggState)))
		elog(ERROR, "sumTopicCout not used as part of an aggregate");

	int32 num_topics = PG_GETARG_INT32(2);

	Datum * array;
	ArrayType * arr_state;
	if (PG_ARGISNULL(0)) {
		array = palloc0(num_topics * sizeof(Datum));
		arr_state = construct_array(array, num_topics, INT4OID, 4, true, 'i');
	} else {
		arr_state = PG_GETARG_ARRAYTYPE_P(0);
	}

	ArrayType * arr_topics = PG_GETARG_ARRAYTYPE_P(1);
	int32 * state = (int32 *)ARR_DATA_PTR(arr_state);
	int32 * topics = (int32 *)ARR_DATA_PTR(arr_topics);

	int32 count = ARR_DIMS(arr_topics)[0];
	for(int32 i = 0; i < count; i++) {
		int32 topic = topics[i];
		state[topic] += 1;
	}

	PG_RETURN_ARRAYTYPE_P(arr_state);
}
