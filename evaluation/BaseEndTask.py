from iterators.DataIterator import DataIterator


class BaseEndTask:
    """Base class for evaluations.

    This class provids a general framework for evaluations.

    General procedures for using these classes:
    1. Instantiate task object.
    2. Add in indices of words in evaluation
        This is typically done by having each DatasetIterator call modify_data for each task.
    3. Evaluate the model
        This is typically called in the main method after the model is trained.
    """

    def __init__(self, args):
        self.args = args

    def modify_data(self, word2id, word_counts):
        """Adds word index/frquency information from the training data.

        Arguments:
            word2id: dict[str, int], map from words to their indices
            word_counts: dict[str, int], map from words to their frequency
        """
        raise NotImplementedError

    def evaluate(self, sess, model):
        """Evaluates a model on the task.

        Arguments:
            sess: tf session
            model: models.BaseModel, model to evaluate
        """
        raise NotImplementedError