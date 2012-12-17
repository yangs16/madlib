/* ----------------------------------------------------------------------- *//**
 *
 * @file newplda.cpp
 *
 * @brief Functions for Parallel Latent Dirichlet Allocation
 *
 * @date Dec 13, 2012
 *//* ----------------------------------------------------------------------- */


#include <dbconnector/dbconnector.hpp>
#include <math.h>
#include "newplda.hpp"

namespace madlib {
namespace modules {
namespace newplda {

using madlib::dbconnector::postgres::madlib_get_typlenbyvalalign;
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
type_info INT4TI(INT4OID);

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
 **/
static int32_t __newplda_gibbs_sample(
    int32_t topic_num, int32_t topic, int32_t * count_d_z, int32_t * count_w_z,
    int32_t * count_z, float8 alpha, float8 beta) 
{
    /* The cumulative probability distribution of the topics */
    double * topic_prs = new double[topic_num]; 

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
    float8 r = drand48();
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
 * @brief This function predicts the topics of words in a document given the
 * learned topic models. The learned topic modesl are passed to this function
 * in the first call and then transfered to the rest calls through
 * args.mSysInfo->user_fctx for efficiency. 
 * @param args[0]           The word count in the document
 * @param args[1]           The unique words in the documents
 * @param args[2]           The counts of each unique words
 * @param args[3]           The topic counts in the document
 * @param args[4]           The topic assignments in the document
 * @param args[5]           The word topic counts, not null for the first call
 *                          in each segment, but null for the rest calls for
 *                          efficiency, refer to the tricks in the join
 *                          operation in the sql calling this function
 * @param args[6]           The corpus topic counts, not null for the first
 *                          call in each segment, but null for the rest calls
 *                          for efficiency, refer to the tricks in the join
 *                          operation in the sql calling this function
 * @param args[7]           The Dirichlet parameter for per-document topic
 *                          multinomial, i.e. alpha
 * @param args[8]           The Dirichlet parameter for per-topic word
 *                          multinomial, i.e. beta
 * @param args[9]           The size of vocabulary
 * @param args[10]           The number of topics
 * @param args[11]          The nunber of iterations (e.g. 20)
 * @return                  The predicted topic counts and topic assignments for
 *                          the document
 **/
AnyType newplda_gibbs_pred::run(AnyType & args)
{
    int32_t word_count = args[0].getAs<int32_t>();
    ArrayHandle<int32_t> words = args[1].getAs<ArrayHandle<int32_t> >();
    ArrayHandle<int32_t> counts = args[2].getAs<ArrayHandle<int32_t> >();
    ArrayHandle<int32_t> topic_count = args[3].getAs<ArrayHandle<int32_t> >();
    ArrayHandle<int32_t> topic_assignment = args[4].getAs<ArrayHandle<int32_t> >();

    double alpha = args[7].getAs<double>();
    double beta = args[8].getAs<double>();
    int32_t voc_size = args[9].getAs<int32_t>();
    int32_t topic_num = args[10].getAs<int32_t>();
    int32_t iter_num = args[11].getAs<int32_t>();

    int32_t __state_size = (voc_size + 1) * topic_num;
    if (!args.getSysInfo()->user_fctx)
    {
        if(args[5].isNull() || args[6].isNull()){
            throw std::domain_error(
                "The parameters word_topic and corpus_topic should not be \
                null for the first call of newplda_gibbs_pred"); 
        }
        ArrayHandle<int32_t> word_topic = args[5].getAs<ArrayHandle<int32_t> >();
        ArrayHandle<int32_t> corpus_topic = args[6].getAs<ArrayHandle<int32_t> >();

        args.getSysInfo()->user_fctx =
            MemoryContextAllocZero(
                    args.getSysInfo()->cacheContext,
                    __state_size * sizeof(int32_t));

        int32_t * state = (int32_t *) args.getSysInfo()->user_fctx;
        memcpy(state, word_topic.ptr(), (voc_size * topic_num) * sizeof(int32_t));
        memcpy(state + voc_size * topic_num, corpus_topic.ptr(), topic_num * sizeof(int32_t));
    }
    int32_t * state = (int32_t *) args.getSysInfo()->user_fctx;
    if(NULL == state){
        throw std::runtime_error("The args.mSysInfo->user_fctx is null.");
    }

    MutableArrayHandle<int32_t> outarray_topic_count(
        construct_array(
            NULL, topic_num, INT4TI.oid, INT4TI.len, INT4TI.byval,
            INT4TI.align));
    memcpy(
        outarray_topic_count.ptr(), topic_count.ptr(), topic_num *
        sizeof(int32_t));

    MutableArrayHandle<int32_t> outarray_topic_assignment(
        construct_array(
            NULL, word_count, INT4TI.oid, INT4TI.len, INT4TI.byval,
            INT4TI.align));
    memcpy(
        outarray_topic_assignment.ptr(), topic_assignment.ptr(), word_count *
        sizeof(int32_t));

    int32_t unique_word_count = words.size();
    for(int it = 0; it < iter_num; it++){
        int32_t word_index = 0;
        for(int32_t i = 0; i < unique_word_count; i++) {
            int32_t wordid = words[i];
            for(int32_t j = 0; j < counts[i]; j++){
                int32_t topic = outarray_topic_assignment[word_index];
                int32_t retopic = __newplda_gibbs_sample(
                    topic_num, topic, outarray_topic_count.ptr(), state +
                    wordid * topic_num, state + voc_size * topic_num, alpha,
                    beta);

                outarray_topic_assignment[word_index] = retopic;
                outarray_topic_count[topic]--;
                outarray_topic_count[retopic]++;
                word_index++;
            }
        }
    }

    AnyType tuple;
    tuple << outarray_topic_count << outarray_topic_assignment;
    return tuple;
}

/**
 * @brief This function learns the topics of words in a document and is the
 * main step of a Gibbs sampling iteration. The word topic counts and
 * corpus topic counts are passed to this function in the first call and
 * then transfered to the rest calls through args.mSysInfo->user_fctx for
 * efficiency. 
 * @param args[0]           The word count in the document
 * @param args[1]           The unique words in the documents
 * @param args[2]           The counts of each unique words
 * @param args[3]           The topic counts in the document
 * @param args[4]           The topic assignments in the document
 * @param args[5]           The word topic counts, not null for the first call
 *                          in each segment, but null for the rest calls for
 *                          efficiency, refer to the tricks in the join
 *                          operation in the sql calling this function
 * @param args[6]           The corpus topic counts, not null for the first
 *                          call in each segment, but null for the rest calls
 *                          for efficiency, refer to the tricks in the join
 *                          operation in the sql calling this function
 * @param args[7]           The Dirichlet parameter for per-document topic
 *                          multinomial, i.e. alpha
 * @param args[8]           The Dirichlet parameter for per-topic word
 *                          multinomial, i.e. beta
 * @param args[9]           The size of vocabulary
 * @param args[10]          The number of topics
 * @return                  The updated topic counts and topic assignments for
 *                          the document
 **/
AnyType newplda_gibbs_train::run(AnyType & args)
{
    int32_t word_count = args[0].getAs<int32_t>();
    ArrayHandle<int32_t> words = args[1].getAs<ArrayHandle<int32_t> >();
    ArrayHandle<int32_t> counts = args[2].getAs<ArrayHandle<int32_t> >();
    ArrayHandle<int32_t> topic_count = args[3].getAs<ArrayHandle<int32_t> >();
    ArrayHandle<int32_t> topic_assignment = args[4].getAs<ArrayHandle<int32_t> >();

    double alpha = args[7].getAs<double>();
    double beta = args[8].getAs<double>();
    int32_t voc_size = args[9].getAs<int32_t>();
    int32_t topic_num = args[10].getAs<int32_t>();

    int32_t __state_size = (voc_size + 1) * topic_num;
    if (!args.getSysInfo()->user_fctx)
    {
        if(args[5].isNull() || args[6].isNull()){
            throw std::domain_error(
                "The parameters word_topic and corpus_topic should not be null \
                for the first call of newplda_gibbs_train"); 
        }
        ArrayHandle<int32_t> word_topic = args[5].getAs<ArrayHandle<int32_t> >();
        ArrayHandle<int32_t> corpus_topic = args[6].getAs<ArrayHandle<int32_t> >();

        args.getSysInfo()->user_fctx =
            MemoryContextAllocZero(
                    args.getSysInfo()->cacheContext,
                    __state_size * sizeof(int32_t));

        int32_t * state = (int32_t *) args.getSysInfo()->user_fctx;
        memcpy(state, word_topic.ptr(), (voc_size * topic_num) * sizeof(int32_t));
        memcpy(state + voc_size * topic_num, corpus_topic.ptr(), topic_num * sizeof(int32_t));
    }

    int32_t * state = (int32_t *) args.getSysInfo()->user_fctx;
    if(NULL == state){
        throw std::runtime_error("The args.mSysInfo->user_fctx is null.");
    }

    MutableArrayHandle<int32_t> outarray_topic_count(
        construct_array(
            NULL, topic_num, INT4TI.oid, INT4TI.len, INT4TI.byval,
            INT4TI.align)); 
    memcpy(
        outarray_topic_count.ptr(), topic_count.ptr(), topic_num *
        sizeof(int32_t));

    MutableArrayHandle<int32_t> outarray_topic_assignment(
        construct_array(
            NULL, word_count, INT4TI.oid, INT4TI.len, INT4TI.byval,
            INT4TI.align));
    memcpy(
        outarray_topic_assignment.ptr(), topic_assignment.ptr(), word_count *
        sizeof(int32_t));

    int32_t unique_word_count = words.size();
    int32_t word_index = 0;
    for(int32_t i = 0; i < unique_word_count; i++) {
        int32_t wordid = words[i];
        for(int32_t j = 0; j < counts[i]; j++){
            int32_t topic = outarray_topic_assignment[word_index];
            int32_t retopic = __newplda_gibbs_sample(
                topic_num, topic, outarray_topic_count.ptr(), state + wordid *
                topic_num, state + voc_size * topic_num, alpha, beta);
            outarray_topic_assignment[word_index] = retopic;
            outarray_topic_count[topic]--;
            outarray_topic_count[retopic]++;

            state[voc_size * topic_num + topic]--;
            state[voc_size * topic_num + retopic]++;
            state[wordid * topic_num + topic]--;
            state[wordid * topic_num + retopic]++;
        }
    }

    AnyType tuple;
    tuple << outarray_topic_count << outarray_topic_assignment;
    return tuple;
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
        throw std::invalid_argument(
            "The word count should be positive integer.");
    if(topic_num < 1)
        throw std::invalid_argument(
            "The topic number should be positive integer.");

    MutableArrayHandle<int32_t> outarray_topic_count(
        construct_array(
            NULL, topic_num, INT4TI.oid, INT4TI.len, INT4TI.byval,
            INT4TI.align));

    MutableArrayHandle<int32_t> outarray_topic_assignment(
        construct_array(
            NULL, word_count, INT4TI.oid, INT4TI.len, INT4TI.byval,
            INT4TI.align));

    for(int32_t i = 0; i < word_count; i++){
        int32_t topic = rand() % topic_num;
        outarray_topic_count[topic] += 1;
        outarray_topic_assignment[i] = topic;  
    }

    AnyType tuple;
    tuple << outarray_topic_count << outarray_topic_assignment;
    return tuple;
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
    int32_t voc_size = args[4].getAs<int32_t>();
    int32_t topic_num = args[5].getAs<int32_t>();

    MutableArrayHandle<int32_t> state(NULL);
    if(args[0].isNull()){
        int dims[2] = {voc_size + 1, topic_num};
        int lbs[2] = {1, 1};
        state = construct_md_array(
            NULL, NULL, 2, dims, lbs, INT4TI.oid, INT4TI.len, INT4TI.byval,
            INT4TI.align);
    } else {
        state = args[0].getAs<MutableArrayHandle<int32_t> >();
    }

    ArrayHandle<int32_t> words = args[1].getAs<ArrayHandle<int32_t> >();
    ArrayHandle<int32_t> counts = args[2].getAs<ArrayHandle<int32_t> >();
    ArrayHandle<int32_t> topics = args[3].getAs<ArrayHandle<int32_t> >();

    int32_t unique_word_count = words.size();
    int32_t word_index = 0;
    for(int32_t i = 0; i < unique_word_count; i++){
        int32_t wordid = words[i];
        for(int32_t j = 0; j < counts[i]; j++){
            int32_t topic = topics[word_index];
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
 * @param args[1]   The state variable, local word topic counts
 * @return          The merged state, element-wise sum of two local states
 **/
AnyType newplda_count_topic_prefunc::run(AnyType & args)
{
    MutableArrayHandle<int32_t> inarray1 = args[0].getAs<MutableArrayHandle<int32_t> >();
    ArrayHandle<int32_t> inarray2 = args[1].getAs<ArrayHandle<int32_t> >();

    if(inarray1.size() != inarray2.size())
        throw std::invalid_argument("Invalid dimension.");

    for(uint32_t i = 0; i < inarray1.size(); i++)
        inarray1[i] += inarray2[i];
    
    return inarray1;
}

AnyType newplda_count_topic_ffunc::run(AnyType & args)
{
    ArrayHandle<int32_t> state = args[0].getAs<ArrayHandle<int32_t> >();
    int32_t ndims = state.dims();
    if(ndims != 2)
        throw std::runtime_error("Invalid state.");

    int32_t voc_size = state.sizeOfDim(0) - 1;
    int32_t topic_num = state.sizeOfDim(1);
        
    int dims[2] = {voc_size, topic_num};
    int lbs[2] = {1, 1};

    MutableArrayHandle<int32_t> word_topic(
        construct_md_array(
            NULL, NULL, 2, dims, lbs, INT4TI.oid, INT4TI.len, INT4TI.byval,
            INT4TI.align));
    memcpy(
        word_topic.ptr(), state.ptr(), voc_size * topic_num * sizeof(int32_t));

    MutableArrayHandle<int32_t> corpus_topic(
        construct_array(
            NULL, state.sizeOfDim(1), INT4TI.oid, INT4TI.len, INT4TI.byval,
            INT4TI.align));
    memcpy(
        corpus_topic.ptr(), state.ptr() + voc_size * topic_num, topic_num *
        sizeof(int32_t));

    AnyType tuple;
    tuple << word_topic << corpus_topic; 
    return tuple;
}

/**
 * @breif This function is the sfunc of the aggregator getting the first value
 * of a column in a segment.
 * @param arg[0]    State variable, null or the first value
 * @param arg[1]    Value in the column in a row
 * @return          Updated state 
 **/

AnyType newplda_first::run(AnyType & args)
{
    if (args[0].isNull())
        return args[1];
    else
        return args[0];
}

AnyType newplda_transpose::run(AnyType & args)
{
    
    if(args[0].isNull())
        throw std::invalid_argument("Null input.");
    
    ArrayHandle<int32_t> matrix = args[0].getAs<ArrayHandle<int32_t> >();
    if(matrix.dims() != 2)
        throw std::domain_error("Invalid dimension.");

    int32_t row_num  = matrix.sizeOfDim(0);
    int32_t col_num  = matrix.sizeOfDim(1);
        
    int dims[2] = {col_num, row_num};
    int lbs[2] = {1, 1};
    MutableArrayHandle<int32_t> transposed(
        construct_md_array(
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
    sr_ctx * ctx = new sr_ctx;

    ArrayHandle<int32_t> inarray = args[0].getAs<ArrayHandle<int32_t> >();
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
        construct_array(
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
