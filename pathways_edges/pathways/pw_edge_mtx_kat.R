

library(dplyr)
ncbi_pathway<-read.table("/data4/petersoa/pathways/NCBI2Reactome_All_Levels.txt", comment.char="", sep="\t")
path1<-"/data4/petersoa/pathways/pw_edge_mtx/"

ncbi_pathway%>%
  mutate(V1=as.character(V1))%>%
  subset(V6=="Homo sapiens")->ncbi_human

edge<-read.csv("/data4/petersoa/pathways/reactome_human_pathway_edges.csv")

edge%>%
  dplyr::select(src, dest, direction, type)%>%
  mutate(src=as.character(src))->edge1

pw<-unique(ncbi_human$V2)

for (j in pw){
  ncbi_human%>%
    subset(V2==j)%>%
    dplyr::select(V1, V2, V4)->ab
  
  ab<-as.data.frame(ab[!duplicated(ab),])
  
  ab_edge<-inner_join(edge1, ab, by=c("src"="V1"))
  
  size_pw<-dim(ab)[1]
  
  if (size_pw>1){
    pw_mtx<-matrix(NA, size_pw, size_pw)
    
    rownames(pw_mtx)<-paste0("entrez_", ab$V1)
    colnames(pw_mtx)<-paste0("entrez_", ab$V1)
    
    if(nrow(ab_edge)!=0){
      ab_edge$src2<-paste0("entrez_", ab_edge$src)
      ab_edge$dest2<-paste0("entrez_", ab_edge$dest)
      
      
      for (i in seq(nrow(ab_edge))){
        row_i<-ab_edge[i,]
        if(row_i$dest2 %in% colnames(pw_mtx)){
          
          if (row_i$direction=="directed"){
            pw_mtx[row_i$src2, row_i$dest2]=1
          }
          else if (row_i$direction=="undirected"){
            pw_mtx[row_i$src2, row_i$dest2]=2
          }
        }
      }
      
      
      file1<-paste0(paste0("pw_mtx_", row_i$V2), ".csv")
      filename<-paste0(path1, file1)
      
      write.csv(pw_mtx, filename)
    }}
  
}
