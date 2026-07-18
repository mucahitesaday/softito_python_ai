"""DFS, YARN ve Spark kavramlarının küçük, çalıştırılabilir benzetimi."""
from dataclasses import dataclass
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parent; RESULTS=ROOT/"results"
class MiniDFS:
    def __init__(self,nodes=3,replication=2): self.nodes=[[] for _ in range(nodes)]; self.replication=replication
    def put(self,name,records,block_size=3):
        blocks=[records[i:i+block_size] for i in range(0,len(records),block_size)]
        for block_id,block in enumerate(blocks):
            for copy in range(self.replication): self.nodes[(block_id+copy)%len(self.nodes)].append({"file":name,"block":block_id,"records":block})
        return len(blocks)
@dataclass
class Job: name:str; cpu:int; memory:int
class MiniYARN:
    def __init__(self,cpu=8,memory=16): self.cpu=cpu; self.memory=memory
    def schedule(self,jobs):
        accepted=[]; cpu=memory=0
        for job in jobs:
            if cpu+job.cpu<=self.cpu and memory+job.memory<=self.memory: accepted.append(job.name); cpu+=job.cpu; memory+=job.memory
        return accepted
class MiniSpark:
    @staticmethod
    def word_count(lines):
        counts={}
        for word in (w.lower() for line in lines for w in line.split()): counts[word]=counts.get(word,0)+1
        return dict(sorted(counts.items(),key=lambda item:(-item[1],item[0])))
def main():
    lines=["python veri yapay zeka","veri büyük veri","python spark veri"]; dfs=MiniDFS(); block_count=dfs.put("lessons.txt",lines,1); yarn=MiniYARN(); scheduled=yarn.schedule([Job("etl",2,4),Job("model",4,8),Job("report",2,3),Job("oversized",4,8)]); result={"dfs_blocks":block_count,"node_block_counts":[len(x) for x in dfs.nodes],"scheduled_jobs":scheduled,"word_count":MiniSpark.word_count(lines)}; RESULTS.mkdir(exist_ok=True); (RESULTS/"cluster_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8"); print("BIG DATA BENZETİMİ\n"+"="*25); print(result)
if __name__=="__main__": main()
