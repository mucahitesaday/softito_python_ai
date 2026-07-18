import unittest
import numpy as np
from DeepLearning.deep_learning_basics import veri_uret, egit
from DeepLearning.sequence_models import simple_rnn, scaled_dot_attention
from ComputerVision.image_processing import synthetic_image, classification_features
from NLP.text_analysis import clean, cooccurrence_embeddings
from LLM_RAG.rag_pipeline import VectorStore, DOCUMENTS, quantize
from MLOps.etl_pipeline import extract, transform
from BigData.mini_cluster import MiniSpark

class FinalTopicsTest(unittest.TestCase):
    def test_ann(self):
        x,y=veri_uret(250); self.assertGreater(egit(x,y,epochs=500)["accuracy"],.7)
    def test_sequence(self):
        states=simple_rnn([.1,.2,.3]); _,weights=scaled_dot_attention(states); self.assertTrue(np.allclose(weights.sum(axis=1),1))
    def test_vision(self): self.assertIn("edge_density",classification_features(synthetic_image()))
    def test_nlp(self):
        vocab,emb=cooccurrence_embeddings(["veri bilimi","veri analizi"]); self.assertEqual(len(vocab),len(emb)); self.assertEqual(clean("BÜYÜK Veri!"),"büyük veri")
    def test_rag(self):
        self.assertGreater(VectorStore(DOCUMENTS).search("vektör arama")[0]["score"],0); self.assertEqual(quantize(np.array([1.]))[0].dtype,np.int8)
    def test_etl(self): self.assertEqual(transform(extract()).isna().sum().sum(),0)
    def test_big_data(self): self.assertEqual(MiniSpark.word_count(["veri veri"])["veri"],2)
if __name__=="__main__": unittest.main()
