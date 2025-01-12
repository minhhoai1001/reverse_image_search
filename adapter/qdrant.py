import qdrant_client.models as models
from qdrant_client import QdrantClient

class QdrantAdapter():
    def __init__(self, host:str="localhost:6333") -> None:
        self.host = host
        try:
            self.client = QdrantClient(host)
        except:
            print('error connection database')
            self.client = None

    def create_collection(self, collection_name, size, hybird=False):
        info = self.client.collection_exists(collection_name=f"{collection_name}")
        if info:
            return None
        
        if hybird:
            info = self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": models.VectorParams(size=size, distance=models.Distance.COSINE),
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(),
                },
            )
        else:
            info = self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE),
            )

        return info
    
    def upsert_points(self, collection_name:str, id:str, payload:dict, feature):
        vector_data = models.PointStruct(
            id=id,
            vector=feature,
            payload=payload,
        )
        info = self.client.upsert(
            collection_name=collection_name,
            wait=True,
            points=[vector_data],
        )

        return info

    def delete_points(self, collection_name:str, ids:list):
        info = self.client.delete(
                        collection_name=collection_name,
                        wait=True,
                        points_selector=models.PointIdsList(points=ids)
                    )

        return info

    def retrieve_ids(self, collection_name:str, ids:list):
        info = self.client.retrieve(
            collection_name=collection_name,
            ids=ids,
            with_vectors=True
        )

        return info

    def search_similar(self, collection_name, vector, limit=5):
        info = self.client.search(
            collection_name=collection_name, 
            query_vector=vector, 
            limit=limit,
            with_payload=True,
        )
        if info:
            return info
        return None
    
    def query_points(self, collection_name, embed, limit=5):
        dense = embed["dense"]
        sparse = embed["sparse"]
        hybrid = self.client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=models.SparseVector(indices=sparse["indices"], values=sparse["values"]),
                    using="sparse",
                    limit=limit,
                ),
                models.Prefetch(
                    query=dense,  # <-- dense vector
                    using="dense",
                    limit=limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
        )
        
        if hybrid:
            return hybrid.points
        else:
            return None