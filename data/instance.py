import copy
import json


class InputInstance(object):
    '''
        use to store a piece of data
        contains:
            idx:    index of this instance
            text_a: first sentence
            text_b: second sentence (if agnews etc. : default None)
            label:  label (if test set: None)
    '''
    def __init__(self, idx, text_a, text_b=None, label=None):
        self.idx = idx
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def perturbable_sentence(self):
        if self.text_b is None:
            return self.text_a
        else:
            return self.text_b

    def is_nli(self):
        return self.text_b is not None

    def length(self):
        return len(self.perturbable_sentence().split())

    @classmethod
    def create_instance_with_perturbed_sentence(cls, instance: "InputInstance", perturb_sent: str):
        idx = instance.idx
        label = instance.label
        if instance.text_b is None:
            text_a = perturb_sent
            text_b = None
        else:
            text_a = instance.text_a
            text_b = perturb_sent
        return cls(idx, text_a, text_b, label)


# if __name__ == '__main__':
#     a = InputInstance(0, 'today is a sunny day.', 'today is a rainy day', 'contradict')
#     b = InputInstance.create_instance_with_perturbed_sentence(a, 'cnmcnmcnm')
#     from transformers import AutoTokenizer
#     c = AutoTokenizer.from_pretrained('textattack/bert-base-uncased-SST-2')
#     print(c.encode_plus([[a.text_a, a.text_b],[b.text_a, b.text_b]]))

