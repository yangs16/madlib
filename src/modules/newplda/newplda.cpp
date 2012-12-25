/* ----------------------------------------------------------------------- *//**
 *
 * @file newplda.cpp
 *
 * @brief Functions for Latent Dirichlet Allocation
 *
 * @date Dec 13, 2012
 *//* ----------------------------------------------------------------------- */


#include <dbconnector/dbconnector.hpp>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <functional>
#include <numeric>
#include "newplda.hpp"

namespace madlib {
namespace modules {
namespace newplda {

using madlib::dbconnector::postgres::madlib_get_typlenbyvalalign;
using madlib::dbconnector::postgres::madlib_construct_array;
using madlib::dbconnector::postgres::madlib_construct_md_array;

typedef struct __type_info{
    Oid oid;
    int16_t len;
    bool    byval;
    char    align;

    __type_info(Oid oid):oid(oid)
    {
        madlib_get_typlenbyvalalign(oid, &len, &byval, &align);
    }
} type_info;
static type_info INT4TI(INT4OID);

/**
 * @brief This function samples a new topic for a word in a document based on
 * the topic counts computed on the rest of the corpus. This is the core
 * function in the Gibbs sampling inference algorithm for LDA. 
 * @param topic_num     The number of topics
 * @param topic         The current topic assignment to a word
 * @param count_d_z     The document topic counts
 * @param count_w_z     The word topic counts
 * @param count_z       The corpus topic counts 
 * @param alpha         The Dirichlet parameter for the per-doc topic
 *                      multinomial 
 * @param beta          The Dirichlet parameter for the per-topic word
 *                      multinomial
 * @return retopic      The new topic assignment to the word
 * @note The topic ranges from 0 to topic_num - 1. 
 *
 * @note For the sake of performance, this function will not check the validity
 * of parameters. The caller will ensure that the three pointers all have non-null
 * values and the lengths are the actual lengths of the arrays. And this
 * function is local to this file only, so this function cannot be maliciously
 * called by intruders.
 **/
static int32_t __newplda_gibbs_sample(
    int32_t topic_num, int32_t topic, const int32_t * count_d_z, const int32_t * count_w_z,
    const int32_t * count_z, double alpha, double beta) 
{
    /* The cumulative probability distribution of the topics */
    double * topic_prs = new double[topic_num]; 
    if(!topic_prs)
        throw std::runtime_error("Out of memory.");

    /* Calculate topic (unnormalised) probabilities */
    double total_unpr = 0;
    for (int32_t i = 0; i < topic_num; i++) {
        int32_t nwz = count_w_z[i];
        int32_t ndz = count_d_z[i];
        int32_t nz = count_z[i];

        /* Adjust the counts to exclude current word's contribution */
        if (i == topic) {
            nwz--;
            ndz--;
            nz--;
        }

        /* Compute the probability */
        double unpr = (ndz + alpha) * (nwz + beta) / (nz + topic_num * beta);
        total_unpr += unpr;
        topic_prs[i] = total_unpr;
    }

    /* Normalise the probabilities */
    for (int32_t i = 0; i < topic_num; i++)
        topic_prs[i] /= total_unpr;

    /* Draw a topic at random */
    double r = drand48();
    int32_t retopic = 0;
    while (true) {
        if (retopic == topic_num - 1 || r < topic_prs[retopic])
            break;
        retopic++; 
    }

    delete[] topic_prs;
    return retopic;
}

/**
 * @brief Get the min value of an array - for parameter checking
 * @return      The min value
 * @note The caller will ensure that ah is always non-null.
 **/
static int32_t __min(
    ArrayHandle<int32_t> ah, size_t start, size_t len){
    const int32_t * array = ah.ptr() + start;
    return *std::min_element(array, array + len);
}
static int32_t __min(ArrayHandle<int32_t> ah){
    return __min(ah, 0, ah.size());
}

/**
 * @brief Get the max value of an array - for parameter checking
 * @return      The max value
 * @note The caller will ensure that ah is always non-null.
 **/
static int32_t __max(
    ArrayHandle<int32_t> ah, size_t start, size_t len){
    const int32_t * array = ah.ptr() + start;
    return *std::max_element(array, array + len);
}
static int32_t __max(ArrayHandle<int32_t> ah){
    return __max(ah, 0, ah.size());
}

/**
 * @brief Get the sum of an array - for parameter checking
 * @return      The sum
 * @note The caller will ensure that ah is always non-null.
 **/
static int32_t __sum(ArrayHandle<int32_t> ah){
    const int32_t * array = ah.ptr();
    int32_t size = ah.size();
    return std::accumulate(array, array + size, 0);
}

/**
 * @brief This function predicts the topics of words in a document given the
 * learned topic models. The learned topic modesl are passed to this function
 * in the first call and then transfered to the rest calls through
 * args.mSysInfo->user_fctx for efficiency. 
 * @param args[0]   The unique words in the documents
 * @param args[1]   The counts of each unique words
 * @param args[2]   The topic counts and topic assignments in the document
 * @param args[3]   The model (word topic counts and corpus topic
 *                  counts), not null for the first call, null for the
 *                  rest of calls for efficiency
 * @param args[4]   The Dirichlet parameter for per-document topic
 *                  multinomial, i.e. alpha
 * @param args[5]   The Dirichlet parameter for per-topic word
 *                  multinomial, i.e. beta
 * @param args[6]   The size of vocabulary
 * @param args[7]   The number of topics
 * @param args[8]   The number of iterations (e.g. 20)
 * @return          The predicted topic counts and topic assignments for
 *                  the document
 **/
AnyType newplda_gibbs_pred::run(AnyType & args)
{
    ArrayHandle<int32_t> words = args[0].getAs<ArrayHandle<int32_t> >();
    ArrayHandle<int32_t> counts = args[1].getAs<ArrayHandle<int32_t> >();
    MutableArrayHandle<int32_t> doc_topic = args[2].getAs<MutableArrayHandle<int32_t> >();
    double alpha = args[4].getAs<double>();
    double beta = args[5].getAs<double>();
    int32_t voc_size = args[6].getAs<int32_t>();
    int32_t topic_num = args[7].getAs<int32_t>();
    int32_t iter_num = args[8].getAs<int32_t>();

    if(words.size() != counts.size())
        throw std::invalid_argument(
            "Dimensions mismatch: words.size() != counts.size().");
    if(__min(words) < 0 || __max(words) >= voc_size)
        throw std::invalid_argument(
            "Invalid values in words");
    if(__min(counts) <= 0)
        throw std::invalid_argument(
            "Invalid values in counts.");
    int32_t word_count = __sum(counts);

    if(alpha <= 0)
        throw std::invalid_argument("Invalid argument - alpha.");
    if(beta <= 0)
        throw std::invalid_argument("Invalid argument - beta.");
    if(voc_size <= 0)
        throw std::invalid_argument(
            "Invalid argument - voc_size");
    if(topic_num <= 0)
        throw std::invalid_argument(
            "Invalid argument - topic_num");
    if(iter_num <= 0)
        throw std::invalid_argument(
            "Invalid argument - iter_num");
    if(doc_topic.size() != (size_t)(word_count + topic_num))
        throw std::invalid_argument(
            "Invalid dimension - doc_topic.size() != word_count + topic_num");
    if(__min(doc_topic, 0, topic_num) < 0)
        throw std::invalid_argument("Invalid values in topic_count.");
    if(
        __min(doc_topic, topic_num, word_count) < 0 ||
        __max(doc_topic, topic_num, word_count) >= topic_num)
        throw std::invalid_argument( "Invalid values in topic_assignment.");

    int32_t __state_size = (voc_size + 1) * topic_num;
    if (!args.getSysInfo()->user_fctx)
    {
        ArrayHandle<int32_t> model = args[3].getAs<ArrayHandle<int32_t> >();
        if(model.size() != (size_t)(__state_size))
            throw std::invalid_argument(
                "Invalid dimension - model.size() != (voc_size + 1) * topic_num.");
        if(__min(model) < 0)
            throw std::invalid_argument("Invalid topic counts in model.");

        args.getSysInfo()->user_fctx =
            MemoryContextAllocZero(
                    args.getSysInfo()->cacheContext,
                    __state_size * sizeof(int32_t));

        int32_t * state = (int32_t *) args.getSysInfo()->user_fctx;
        memcpy(state, model.ptr(), __state_size * sizeof(int32_t));
    }
    int32_t * state = (int32_t *) args.getSysInfo()->user_fctx;
    if(NULL == state){
        throw std::runtime_error("The args.mSysInfo->user_fctx is null.");
    }

    // iteration is done within this c function to avoid the overhead by
    // multiple UDF calls
    int32_t unique_word_count = words.size();
    for(int it = 0; it < iter_num; it++){
        int32_t word_index = topic_num;
        for(int32_t i = 0; i < unique_word_count; i++) {
            int32_t wordid = words[i];
            for(int32_t j = 0; j < counts[i]; j++){
                int32_t topic = doc_topic[word_index];
                int32_t retopic = __newplda_gibbs_sample(
                    topic_num, topic, doc_topic.ptr(), state +
                    wordid * topic_num, state + voc_size * topic_num, alpha,
                    beta);

                doc_topic[word_index] = retopic;
                doc_topic[topic]--;
                doc_topic[retopic]++;
                word_index++;
            }
        }
    }
    
    return doc_topic;
}

/**
 * @brief This function learns the topics of words in a document and is the
 * main step of a Gibbs sampling iteration. The word topic counts and
 * corpus topic counts are passed to this function in the first call and
 * then transfered to the rest calls through args.mSysInfo->user_fctx for
 * efficiency. 
 * @param args[0]   The unique words in the documents
 * @param args[1]   The counts of each unique words
 * @param args[2]   The topic counts and topic assignments in the document
 * @param args[3]   The model (word topic counts and corpus topic
 *                  counts), not null for the first call, null for the
 *                  rest of calls for efficiency
 * @param args[4]   The Dirichlet parameter for per-document topic
 *                  multinomial, i.e. alpha
 * @param args[5]   The Dirichlet parameter for per-topic word
 *                  multinomial, i.e. beta
 * @param args[6]   The size of vocabulary
 * @param args[7]   The number of topics
 * @return          The updated topic counts and topic assignments for
 *                  the document
 **/
AnyType newplda_gibbs_train::run(AnyType & args)
{
    ArrayHandle<int32_t> words = args[0].getAs<ArrayHandle<int32_t> >();
    ArrayHandle<int32_t> counts = args[1].getAs<ArrayHandle<int32_t> >();
    MutableArrayHandle<int32_t> doc_topic = args[2].getAs<MutableArrayHandle<int32_t> >();
    double alpha = args[4].getAs<double>();
    double beta = args[5].getAs<double>();
    int32_t voc_size = args[6].getAs<int32_t>();
    int32_t topic_num = args[7].getAs<int32_t>();
    
    if(words.size() != counts.size())
        throw std::invalid_argument(
            "Dimensions mismatch - words.size() != counts.size().");
    if(__min(words) < 0 || __max(words) >= voc_size)
        throw std::invalid_argument(
            "Invalid values in words");
    if(__min(counts) <= 0)
        throw std::invalid_argument(
            "Invalid values in counts.");
    int32_t word_count = __sum(counts);

    if(alpha <= 0)
        throw std::invalid_argument("Invalid argument - alpha.");
    if(beta <= 0)
        throw std::invalid_argument("Invalid argument - beta.");
    if(voc_size <= 0)
        throw std::invalid_argument(
            "Invalid argument - voc_size");
    if(topic_num <= 0)
        throw std::invalid_argument(
            "Invalid argument - topic_num");

    if(doc_topic.size() != (size_t)(word_count + topic_num))
        throw std::invalid_argument(
            "Invalid dimension - doc_topic.size() != word_count + topic_num");
    if(__min(doc_topic, 0, topic_num) < 0)
        throw std::invalid_argument("Invalid values in topic_count.");
    if(
        __min(doc_topic, topic_num, word_count) < 0 ||
        __max(doc_topic, topic_num, word_count) >= topic_num)
        throw std::invalid_argument( "Invalid values in topic_assignment.");
    if(doc_topic.size() != (size_t)(word_count + topic_num))
        throw std::invalid_argument(
            "Invalid dimension - topic_count.size() != topic_num.");

    int32_t __state_size = (voc_size + 1) * topic_num;
    if (!args.getSysInfo()->user_fctx)
    {
        ArrayHandle<int32_t> model = args[3].getAs<ArrayHandle<int32_t> >();
        if(model.size() != (size_t)(__state_size))
            throw std::invalid_argument(
                "Invalid dimension - model.size() != (voc_size + 1) * topic_num.");
        if(__min(model) < 0)
            throw std::invalid_argument("Invalid topic counts in model.");

        args.getSysInfo()->user_fctx =
            MemoryContextAllocZero(
                    args.getSysInfo()->cacheContext,
                    __state_size * sizeof(int32_t));

        int32_t * state = (int32_t *) args.getSysInfo()->user_fctx;
        memcpy(state, model.ptr(),  __state_size * sizeof(int32_t));
    }

    int32_t * state = (int32_t *) args.getSysInfo()->user_fctx;
    if(NULL == state){
        throw std::runtime_error("The args.mSysInfo->user_fctx is null.");
    }

    int32_t unique_word_count = words.size();
    int32_t word_index = topic_num;
    for(int32_t i = 0; i < unique_word_count; i++) {
        int32_t wordid = words[i];
        for(int32_t j = 0; j < counts[i]; j++){
            int32_t topic = doc_topic[word_index];
            int32_t retopic = __newplda_gibbs_sample(
                topic_num, topic, doc_topic.ptr(), state + wordid *
                topic_num, state + voc_size * topic_num, alpha, beta);
            doc_topic[word_index] = retopic;
            doc_topic[topic]--;
            doc_topic[retopic]++;

            state[voc_size * topic_num + topic]--;
            state[voc_size * topic_num + retopic]++;
            state[wordid * topic_num + topic]--;
            state[wordid * topic_num + retopic]++;
            word_index++;
        }
    }

    return doc_topic;
}

/**
 * @brief This function assigns topics to words in a document randomly and
 * returns the topic counts and topic assignments.
 * @param args[0]   The word count in the documents
 * @param args[1]   The number of topics
 * @result          The topic counts and topic assignments 
 *                  (length = topic_num + word_count)
 **/
AnyType newplda_random_assign::run(AnyType & args)
{
    int32_t word_count = args[0].getAs<int32_t>();
    int32_t topic_num = args[1].getAs<int32_t>();

    if(word_count < 1)
        throw std::invalid_argument( "Invalid argument - word_count.");
    if(topic_num < 1)
        throw std::invalid_argument( "Invalid argument - topic_num.");

    MutableArrayHandle<int32_t> doc_topic(
        madlib_construct_array(
            NULL, topic_num + word_count, INT4TI.oid, INT4TI.len, INT4TI.byval,
            INT4TI.align));

    for(int32_t i = 0; i < word_count; i++){
        int32_t topic = random() % topic_num;
        doc_topic[topic] += 1;
        doc_topic[topic_num + i] = topic;  
    }

    return doc_topic;
}

/**
 * @brief This function is the sfunc for the aggregator computing the topic
 * counts. It scans the topic assignments in a document and updates the word
 * topic counts.
 * @param args[0]   The state variable, current topic counts
 * @param args[1]   The unique words in the document
 * @param args[2]   The counts of each unique word in the document
 * @param args[3]   The topic assignments in the document
 * @param args[4]   The size of vocabulary
 * @param args[5]   The number of topics 
 * @return          The updated state
 **/
AnyType newplda_count_topic_sfunc::run(AnyType & args)
{
    if(!(args.getFCInfo()->context && IsA(args.getFCInfo()->context, AggState)))
        throw std::runtime_error(
            "This function should be called in an aggregator.");

    if(args[4].isNull() || args[5].isNull())
        throw std::invalid_argument("Null input.");

    if(args[1].isNull() || args[2].isNull() || args[3].isNull()) 
        return args[0];

    int32_t voc_size = args[4].getAs<int32_t>();
    int32_t topic_num = args[5].getAs<int32_t>();
    if(voc_size <= 0)
        throw std::invalid_argument(
            "Invalid argument - voc_size.");
    if(topic_num <= 0)
        throw std::invalid_argument(
            "Invalid argument - topic_num.");

    ArrayHandle<int32_t> words = args[1].getAs<ArrayHandle<int32_t> >();
    ArrayHandle<int32_t> counts = args[2].getAs<ArrayHandle<int32_t> >();
    ArrayHandle<int32_t> topic_assignment = args[3].getAs<ArrayHandle<int32_t> >();
    if(words.size() != counts.size())
        throw std::invalid_argument(
            "Dimensions mismatch - words.size() != counts.size().");
    if(__min(words) < 0 || __max(words) >= voc_size)
        throw std::invalid_argument(
            "Invalid values in words");
    if(__min(counts) <= 0)
        throw std::invalid_argument(
            "Invalid values in counts.");
    if(__min(topic_assignment) < 0 || __max(topic_assignment) >= topic_num)
        throw std::invalid_argument("Invalid values in topics");
    if((size_t)__sum(counts) != topic_assignment.size())
        throw std::invalid_argument(
            "Dimension mismatch - sum(counts) != topic_assignment.size()");

    MutableArrayHandle<int32_t> state(NULL);
    if(args[0].isNull()){
        int dims[2] = {voc_size + 1, topic_num};
        int lbs[2] = {1, 1};
        state = madlib_construct_md_array(
            NULL, NULL, 2, dims, lbs, INT4TI.oid, INT4TI.len, INT4TI.byval,
            INT4TI.align);
    } else {
        state = args[0].getAs<MutableArrayHandle<int32_t> >();
    }

    int32_t unique_word_count = words.size();
    int32_t word_index = 0;
    for(int32_t i = 0; i < unique_word_count; i++){
        int32_t wordid = words[i];
        for(int32_t j = 0; j < counts[i]; j++){
            int32_t topic = topic_assignment[word_index];
            state[wordid * topic_num + topic]++;
            state[voc_size * topic_num + topic]++;
            word_index++;
        }
    }

    return state;
}

/**
 * @brief This function is the prefunc for the aggregator computing the
 * topic counts.
 * @param args[0]   The state variable, local topic counts
 * @param args[1]   The state variable, local topic counts
 * @return          The merged state, element-wise sum of two local states
 **/
AnyType newplda_count_topic_prefunc::run(AnyType & args)
{
    MutableArrayHandle<int32_t> state1 = args[0].getAs<MutableArrayHandle<int32_t> >();
    ArrayHandle<int32_t> state2 = args[1].getAs<ArrayHandle<int32_t> >();

    if(state1.size() != state2.size())
        throw std::invalid_argument("Invalid dimension.");

    for(size_t i = 0; i < state1.size(); i++)
        state1[i] += state2[i];
    
    return state1;
}

AnyType newplda_transpose::run(AnyType & args)
{
    ArrayHandle<int32_t> matrix = args[0].getAs<ArrayHandle<int32_t> >();
    if(matrix.dims() != 2)
        throw std::domain_error("Invalid dimension.");

    int32_t row_num  = matrix.sizeOfDim(0);
    int32_t col_num  = matrix.sizeOfDim(1);
        
    int dims[2] = {col_num, row_num};
    int lbs[2] = {1, 1};
    MutableArrayHandle<int32_t> transposed(
        madlib_construct_md_array(
            NULL, NULL, 2, dims, lbs, INT4TI.oid, INT4TI.len, INT4TI.byval,
            INT4TI.align));

    for(int32_t i = 0; i < row_num; i++){
        int32_t index = i * col_num;
        for(int32_t j = 0; j < col_num; j++){
               transposed[j * row_num + i] = matrix[index]; 
               index++;
        }
    }

    return transposed;
}

typedef struct __sr_ctx{
    const int32_t * inarray;
    int32_t maxcall;
    int32_t dim;
    int32_t curcall;
} sr_ctx;

void * newplda_unnest::SRF_init(AnyType &args) 
{
    ArrayHandle<int32_t> inarray = args[0].getAs<ArrayHandle<int32_t> >();
    if(inarray.dims() != 2)
        throw std::invalid_argument("Invalid dimension.");

    sr_ctx * ctx = new sr_ctx;
    ctx->inarray = inarray.ptr();
    ctx->maxcall = inarray.sizeOfDim(0);
    ctx->dim = inarray.sizeOfDim(1);
    ctx->curcall = 0;

    return ctx;
}

AnyType newplda_unnest::SRF_next(void *user_fctx, bool *is_last_call)
{
    sr_ctx * ctx = (sr_ctx *) user_fctx;
    if (ctx->maxcall == 0) {
        *is_last_call = true;
        return Null();
    }

    MutableArrayHandle<int32_t> outarray(
        madlib_construct_array(
            NULL, ctx->dim, INT4TI.oid, INT4TI.len, INT4TI.byval,
            INT4TI.align));
    memcpy(
        outarray.ptr(), ctx->inarray + ctx->curcall * ctx->dim, ctx->dim *
        sizeof(int32_t));

    ctx->curcall++;
    ctx->maxcall--;
    *is_last_call = false;

    return outarray;
}
}
}
}
