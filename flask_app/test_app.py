from interface import interactive_demo as idemo

from test_attention_only import test


# id = idemo()
# print(id.predict_for_all("this is a sent"))

sentence = "all right, everybody go ahead and have a seat ."
attention_only_output = test(sentence)
output_str_attention_only_output = " ".join(attention_only_output)
print(output_str_attention_only_output)