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
#include <time.h>
     

#ifndef NO_PG_MODULE_MAGIC
PG_MODULE_MAGIC;
#endif
/* Indicate "version 1" calling conventions for all exported functions. */
PG_FUNCTION_INFO_V1(randomAssign);
PG_FUNCTION_INFO_V1(sampleNewTopic);
PG_FUNCTION_INFO_V1(sumTopicCount);
PG_FUNCTION_INFO_V1(intArrayAdd);
PG_FUNCTION_INFO_V1(logLikelihood);

/************************************************************************ 
 * Begin: Experiment with window function to transfer states between rows 
 * Using encoded int32 array should be more efficient 
*************************************************************************/ 
PG_FUNCTION_INFO_V1(sFunc4GibbsWFuncFast);
PG_FUNCTION_INFO_V1(fFunc4GibbsWFuncFast);

Datum fFunc4GibbsWFuncFast(PG_FUNCTION_ARGS);
Datum fFunc4GibbsWFuncFast(PG_FUNCTION_ARGS)
{
	/*
	clock_t start = clock();
	*/
	ArrayType * arr_state = PG_GETARG_ARRAYTYPE_P(0);
	int32 * state = (int32 *)ARR_DATA_PTR(arr_state);

	int32 docid = state[0];
	int32 wordid = state[1];
	int32 tcount = state[2];

	ArrayType * arr_res = construct_array(NULL, (tcount + 2), INT4OID, 4, true, 'i');
	int32 * res = (int32 *)ARR_DATA_PTR(arr_res);
	res[0] = docid;
	res[1] = wordid;
	memcpy(res + 2, state + 3, tcount * sizeof(int32));

	/*
	clock_t end = clock()
	elog(WARNING, "ffunc: %d", end - start);
	*/
	PG_RETURN_ARRAYTYPE_P(arr_res);
}

/**
 * state int4[]
 * docid int4, wordid int4, topics int4[], 
 * count_d_z int4[], count_w_z int4[], count_z int4[], 
 * alpha float, beta float,
 * wcount int4, zcount int4)
**/
static int32 __sampleTopic(int32, int32, int32 *, int32 *, int32 *, float8, float8); 

typedef struct {
	int32 * docid;
	int32 * wordid;
	int32 * tcount;
	int32 * topics;
	int32 * corpus_topic;
	int32 * doc_topic;
	int32 * word_topic;
	int32 * wlist;
} __gibbs_state_fast;

#define TCOUNT 100
Datum sFunc4GibbsWFuncFast(PG_FUNCTION_ARGS);
Datum sFunc4GibbsWFuncFast(PG_FUNCTION_ARGS)
{
	/*
	clock_t start = clock();
	*/
	__gibbs_state_fast gsf;

	int32 docid = PG_GETARG_INT32(1);
	int32 wordid = PG_GETARG_INT32(2);

	ArrayType * arr_topics = PG_GETARG_ARRAYTYPE_P(3);
	ArrayType * arr_c_d_z = PG_GETARG_ARRAYTYPE_P(4);
	ArrayType * arr_c_w_z = PG_GETARG_ARRAYTYPE_P(5);
	ArrayType * arr_c_z = PG_GETARG_ARRAYTYPE_P(6);

 	int32 * topics = (int32 *)ARR_DATA_PTR(arr_topics);
 	int32 * c_d_z = (int32 *)ARR_DATA_PTR(arr_c_d_z);
 	int32 * c_w_z = (int32 *)ARR_DATA_PTR(arr_c_w_z);
 	int32 * c_z = (int32 *)ARR_DATA_PTR(arr_c_z);

	float8 alpha = PG_GETARG_FLOAT8(7);
	float8 beta = PG_GETARG_FLOAT8(8);
	int32 wcount = PG_GETARG_INT32(9);
	int32 zcount = PG_GETARG_INT32(10);
 	int32 tcount  = ARR_DIMS(arr_topics)[0];
	if(tcount > TCOUNT)
		tcount = TCOUNT;
	
	ArrayType * arr_return_state;
	int32 state_array_len = 3 + TCOUNT + 2 * zcount + wcount * zcount + wcount;
	int32 topics_offset = 3;
	int32 corpus_topic_offset = 3 + TCOUNT;
	int32 doc_topic_offset = 3 + TCOUNT + zcount;
	int32 word_topic_offset = 3 + TCOUNT + 2 * zcount;
	int32 wlist_offset = 3 + TCOUNT + 2 * zcount + wcount * zcount; 

	if(PG_ARGISNULL(0)){
		arr_return_state = construct_array(NULL, state_array_len, INT4OID, 4, true, 'i');
		int32 * return_state = (int32 *) ARR_DATA_PTR(arr_return_state);
		
		gsf.docid = return_state;
		gsf.wordid = return_state + 1;
		gsf.tcount = return_state + 2;
		gsf.topics = return_state + topics_offset;
		gsf.corpus_topic = return_state + corpus_topic_offset;
		gsf.doc_topic = return_state + doc_topic_offset;
		gsf.word_topic = return_state + word_topic_offset;
		gsf.wlist = return_state + wlist_offset;

		*(gsf.docid) = docid;
		*(gsf.wordid) = wordid;
		*(gsf.tcount) = tcount;
		memcpy(gsf.topics, topics, tcount * sizeof(int32));
		memcpy(gsf.corpus_topic, c_z, zcount * sizeof(int32));
		memcpy(gsf.doc_topic, c_d_z, zcount * sizeof(int32));
		memcpy(gsf.word_topic + wordid * zcount, c_w_z, zcount * sizeof(int32));
		gsf.wlist[wordid] = 1;
	}else{
		arr_return_state = PG_GETARG_ARRAYTYPE_P(0);
		int32 * return_state = (int32 *)ARR_DATA_PTR(arr_return_state);

		gsf.docid = return_state;
		gsf.wordid = return_state + 1;
		gsf.tcount = return_state + 2;
		gsf.topics = return_state + topics_offset;
		gsf.corpus_topic = return_state + corpus_topic_offset;
		gsf.doc_topic = return_state + doc_topic_offset;
		gsf.word_topic = return_state + word_topic_offset;
		gsf.wlist = return_state + wlist_offset;

		*(gsf.wordid) = wordid;
		*(gsf.tcount) = tcount;
		if (docid != *(gsf.docid)){
			memcpy(gsf.doc_topic, c_d_z, zcount * sizeof(int32));
			*(gsf.docid) = docid;
		}

		if(gsf.wlist[wordid] < 1){
			memcpy(gsf.word_topic + wordid * zcount, c_w_z, zcount * sizeof(int32));
			gsf.wlist[wordid] = 1;
			*(gsf.wordid) = wordid;
		}
	}

	/*
	clock_t ss = clock();
	*/
	int32 * new_topics = (int32 *)palloc(tcount * sizeof(int32));
	for(int32 i = 0; i < tcount; i++) {
		int32 topic = topics[i];
		int32 retopic = __sampleTopic(zcount, topic, gsf.doc_topic, gsf.word_topic + wordid * zcount, gsf.corpus_topic, alpha, beta);
		new_topics[i] = retopic;

		gsf.corpus_topic[topic]--;
		gsf.corpus_topic[retopic]++;
		gsf.doc_topic[topic]--;
		gsf.doc_topic[retopic]++;
		gsf.word_topic[wordid * zcount + topic]--;
		gsf.word_topic[wordid * zcount + retopic]++;
	}

	memcpy(gsf.topics, new_topics, tcount * sizeof(int32));
	pfree(new_topics);
	/*
	clock_t ee = clock();
	elog(WARNING, "\tsampling: %d", ee - ss);
	clock_t end = clock();
	elog(WARNING, "sfunc: %d", end - start);
	*/
	PG_RETURN_ARRAYTYPE_P(arr_return_state);
}
/************************************************************************ 
 * End
 ************************************************************************/ 

/************************************************************************ 
 * Begin: Experiment with window function to transfer states between rows 
 * Using tuple to handle composite type, very slow and inefficient
*************************************************************************/ 
PG_FUNCTION_INFO_V1(sFunc4GibbsWFunc);
PG_FUNCTION_INFO_V1(fFunc4GibbsWFunc);

static int32 __sampleTopic(int32, int32, int32 *, int32 *, int32 *, float8, float8); 

Datum fFunc4GibbsWFunc(PG_FUNCTION_ARGS);
Datum fFunc4GibbsWFunc(PG_FUNCTION_ARGS)
{
	HeapTupleHeader tuphead = PG_GETARG_HEAPTUPLEHEADER(0);
	Datum values[3];
	bool isnull;

	values[0] = GetAttributeByName(tuphead, "docid", &isnull);
	values[1] = GetAttributeByName(tuphead, "wordid", &isnull);
	values[2] = GetAttributeByName(tuphead, "topics", &isnull);

	TupleDesc tupdesc;
	if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
           	ereport(ERROR, (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                     errmsg("function returning record called in context "
                       	    "that cannot accept type record")));
	tupdesc = BlessTupleDesc(tupdesc);

	bool isnulls[3] = {false, false, false};
	HeapTuple heaptup = heap_form_tuple(tupdesc, values, isnulls);
	Datum result = HeapTupleGetDatum(heaptup);
	PG_RETURN_DATUM(result);
}

typedef struct {
	int32 docid;
	/*int32 wordid;*/
	/*int32 * topics;*/
	int32 * c_z;
	int32 * c_d_z;
	int32 * c_w_z;
	int32 * wlist;
	/*ArrayType * arr_topics;*/
	ArrayType * arr_c_z;
	ArrayType * arr_c_d_z;
	ArrayType * arr_c_w_z;
	ArrayType * arr_wlist;
}__gibbs_state;

/**
 * state MADLIB_SCHEMA.__newplda_gibbs_state, 
 * docid int4, wordid int4, topics int4[], 
 * count_d_z int4[], count_w_z int4[], count_z int4[], 
 * alpha float, beta float,
 * wcount int4, zcount int4)
**/

Datum sFunc4GibbsWFunc(PG_FUNCTION_ARGS);
Datum sFunc4GibbsWFunc(PG_FUNCTION_ARGS)
{
	__gibbs_state state;

	int32 docid = PG_GETARG_INT32(1);
	int32 wordid = PG_GETARG_INT32(2);

	ArrayType * arr_topics = PG_GETARG_ARRAYTYPE_P(3);
	ArrayType * arr_c_d_z = PG_GETARG_ARRAYTYPE_P(4);
	ArrayType * arr_c_w_z = PG_GETARG_ARRAYTYPE_P(5);
	ArrayType * arr_c_z = PG_GETARG_ARRAYTYPE_P(6);

 	int32 * topics = (int32 *)ARR_DATA_PTR(arr_topics);
 	int32 * c_d_z = (int32 *)ARR_DATA_PTR(arr_c_d_z);
 	int32 * c_w_z = (int32 *)ARR_DATA_PTR(arr_c_w_z);
 	int32 * c_z = (int32 *)ARR_DATA_PTR(arr_c_z);

	float8 alpha = PG_GETARG_FLOAT8(7);
	float8 beta = PG_GETARG_FLOAT8(8);
	int32 wcount = PG_GETARG_INT32(9);	/* used only once */
	int32 zcount = PG_GETARG_INT32(10);	/* used only once */
 	int32 tcount  = ARR_DIMS(arr_topics)[0];
	
	if(PG_ARGISNULL(0)){
		state.docid = docid;

		ArrayType * arr = construct_array(NULL, zcount, INT4OID, 4, true, 'i');
		state.arr_c_z = arr;
		state.c_z = (int32 *)ARR_DATA_PTR(arr);
		memcpy(state.c_z, c_z, zcount * sizeof(int32));

		arr = construct_array(NULL, zcount, INT4OID, 4, true, 'i');
		state.arr_c_d_z = arr;
		state.c_d_z = (int32 *)ARR_DATA_PTR(arr);
		memcpy(state.c_d_z, c_d_z, zcount * sizeof(int32));

		arr = construct_array(NULL, wcount * zcount, INT4OID, 4, true, 'i');
		state.arr_c_w_z = arr;
		state.c_w_z = (int32 *)ARR_DATA_PTR(arr);
		memcpy(state.c_w_z + wordid * zcount, c_w_z, zcount * sizeof(int32));

		arr = construct_array(NULL, wcount, INT4OID, 4, true, 'i');
		state.arr_wlist = arr;
		state.wlist = (int32 *)ARR_DATA_PTR(arr);
		state.wlist[wordid] = 1;
	}else{
		HeapTupleHeader tuphead = PG_GETARG_HEAPTUPLEHEADER(0);
		bool isnull;

		state.docid = DatumGetInt32(GetAttributeByName(tuphead, "docid", &isnull));
		state.arr_c_z = DatumGetArrayTypeP(GetAttributeByName(tuphead, "corpus_topic", &isnull));
		state.c_z = (int32 *)ARR_DATA_PTR(state.arr_c_z);

		state.arr_c_d_z = DatumGetArrayTypeP(GetAttributeByName(tuphead, "doc_topic", &isnull));
		state.c_d_z = (int32 *)ARR_DATA_PTR(state.arr_c_d_z);
		if (docid != state.docid){
			memcpy(state.c_d_z, c_d_z, zcount * sizeof(int32));
			state.docid = docid;
		}
	
		state.arr_wlist = DatumGetArrayTypeP(GetAttributeByName(tuphead, "word_list", &isnull));
		state.wlist = (int32 *)ARR_DATA_PTR(state.arr_wlist);

		state.arr_c_w_z = DatumGetArrayTypeP(GetAttributeByName(tuphead, "word_topic", &isnull));
		state.c_w_z = (int32 *)ARR_DATA_PTR(state.arr_c_w_z);
		if(state.wlist[wordid] < 1){
			memcpy(state.c_w_z + wordid * zcount, c_w_z, zcount * sizeof(int32));
			state.wlist[wordid] = 1;
		}
	}

	ArrayType * arr_result_topics = construct_array(NULL, tcount, INT4OID, 4, true, 'i');
	int32 * result_topics = (int32 *)ARR_DATA_PTR(arr_result_topics);
	for(int32 i = 0; i < tcount; i++) {
		int32 topic = topics[i];
		int32 retopic = __sampleTopic(zcount, topic, state.c_d_z, state.c_w_z + wordid * zcount, state.c_z, alpha, beta);
		result_topics[i] = retopic;

		state.c_z[topic]--;
		state.c_z[retopic]++;
		state.c_d_z[topic]--;
		state.c_d_z[retopic]++;
		state.c_w_z[wordid * zcount + topic]--;
		state.c_w_z[wordid * zcount + retopic]++;
	}

	TupleDesc tupdesc;
	if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
		ereport(ERROR, (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			errmsg("function returning record called in context "
                        	    "that cannot accept type record")));
	tupdesc= BlessTupleDesc(tupdesc);

	Datum values[7];
	values[0] = Int32GetDatum(state.docid);
	values[1] = Int32GetDatum(wordid);
	values[2] = PointerGetDatum(arr_result_topics);	
	values[3] = PointerGetDatum(state.arr_c_z);	
	values[4] = PointerGetDatum(state.arr_c_d_z);	
	values[5] = PointerGetDatum(state.arr_c_w_z);	
	values[6] = PointerGetDatum(state.arr_wlist);	

	bool isnulls[7] = {false, false, false, false, false, false, false};
	HeapTuple heaptup = heap_form_tuple(tupdesc, values, isnulls);
	Datum result = HeapTupleGetDatum(heaptup);
	PG_RETURN_DATUM(result);
}
/************************************************************************ 
 * End
 ************************************************************************/ 

static float8 __logBeta(float8 alpha, int32 len){
	if (alpha <= 0 || len <= 0)
		elog(ERROR, "Invalid parameter alpha and len for log_beta: %.f\t %d", alpha, len);
		
	return len * lgamma(alpha) - lgamma(len * alpha);
}

static float8 __logMultiBeta(float8 * pTopics, int32 len){
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
	int32 wcount = PG_GETARG_INT32(0);
	int32 zcount = PG_GETARG_INT32(1);

	if(wcount < 0 || zcount < 0)
		elog(ERROR, "Word count or topic number should be no less than 1.");

	ArrayType * arr = construct_array(NULL, wcount, INT4OID, 4, true, 'i');
	int32 * result = (int32 *)ARR_DATA_PTR(arr);

	for(int32 i = 0; i < wcount; i++)
		result[i] = rand() % zcount;

	PG_RETURN_ARRAYTYPE_P(arr);
}

Datum logLikelihood(PG_FUNCTION_ARGS);
Datum logLikelihood(PG_FUNCTION_ARGS)
{
	float8 ll = PG_GETARG_FLOAT8(0);
	float8 hyper = PG_GETARG_FLOAT8(2);

	ArrayType * arr = PG_GETARG_ARRAYTYPE_P(1);
	int32 * topics = (int32 *)ARR_DATA_PTR(arr);
	int32 zcount = ARR_DIMS(arr)[0];

	float8 * tmp = (float8 *)palloc(sizeof(float8) * zcount); 
	for(int32 i = 0; i < zcount; i ++)
		tmp[i] = topics[i] + hyper;

	ll += __logMultiBeta(tmp, zcount);
	ll -= __logBeta(hyper, zcount);

	pfree(tmp);
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
 *  @param beta		the Dirichlet parameter for the per-topic word multinomial
 *
 * The function is non-destructive to all the input arguments.
 */
static int32 __sampleTopic(int32 num_topics, int32 topic, int32 * count_d_z, int32 * count_w_z, int32 * count_z, float8 alpha, float8 beta) 
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
	float8 beta = PG_GETARG_FLOAT8(5);

	ArrayType * ret_topics_arr;
	int32 * ret_topics;

	int32 count = ARR_DIMS(arr_topics)[0];
	ret_topics_arr = construct_array(NULL, count, INT4OID, 4, true, 'i');
	ret_topics = (int32 *)ARR_DATA_PTR(ret_topics_arr);

	for(int32 i = 0; i < count; i++) {
		int32 topic = topics[i];
		int32 rtopic = __sampleTopic(num_topics, topic, count_d_z, count_w_z, count_z, alpha, beta);
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

	ArrayType * arr_state;
	int32 * state;
	if (PG_ARGISNULL(0)) {
		arr_state = construct_array(NULL, num_topics, INT4OID, 4, true, 'i');
		state = (int32 *)ARR_DATA_PTR(arr_state);
	} else {
		arr_state = PG_GETARG_ARRAYTYPE_P(0);
		state = (int32 *)ARR_DATA_PTR(arr_state);
	}

	ArrayType * arr_topics = PG_GETARG_ARRAYTYPE_P(1);
	int32 * topics = (int32 *)ARR_DATA_PTR(arr_topics);

	int32 count = ARR_DIMS(arr_topics)[0];
	for(int32 i = 0; i < count; i++) {
		int32 topic = topics[i];
		state[topic] += 1;
	}

	PG_RETURN_ARRAYTYPE_P(arr_state);
}

Datum intArrayAdd(PG_FUNCTION_ARGS);
Datum intArrayAdd(PG_FUNCTION_ARGS)
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
