library(graphite)
library(dplyr)

edges1<-as.data.frame(matrix(ncol=7))

colnames(edges1)<-c("src_type", "src", "dest_type", "dest", "direction", "type", "pathway_id")

paths <- graphite::pathways('hsapiens','reactome')
paths[[1]]

c<-paths@entries

print("yes")

for(i in c){
  a<-convertIdentifiers(i, "entrez")
  
  edges<-a@protEdges
  
  if( dim(edges)[1]>0){
    
    edges$pathway_id<-a@id
    
    edges1<-rbind(edges1, edges)
  }
}

write.csv(edges1, "/data4/petersoa/pathways/reactome_human_pathway_edges.csv")