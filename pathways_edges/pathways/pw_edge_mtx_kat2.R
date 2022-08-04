

library(dplyr)
ncbi_pathway<-read.table("/data3/darpa/omics_databases/ensembl2pathway/pathways_edges/pathways/NCBI2Reactome_All_Levels.txt", header=FALSE )

cat(dim(ncbi_pathway))

ncbi_pathway%>%
  subset(startsWith(V2, "R-HSA"))%>%
  mutate(V1=as.character(V1))->ncbi_human


path1<-"/data3/darpa/omics_databases/ensembl2pathway/pathways_edges/pathways/pw_edge_mtx/"


edge<-read.csv("/data3/darpa/omics_databases/ensembl2pathway/pathways_edges/pathways/reactome_human_pathway_edges.csv")

edge%>%
  dplyr::select(src, dest, direction, type)%>%
  mutate(src=as.character(src))->edge1

pw<-unique(ncbi_human$V2)

cat("\n num pws ")
cat(dim(pw))
cat("\n")

for (j in pw){
print(j)
  ncbi_human%>%
    subset(V2==j)%>%
    dplyr::select(V1, V2)->ab
  
  ab<-as.data.frame(ab[!duplicated(ab$V1),])
  
  ab_edge<-inner_join(edge1, ab, by=c("src"="V1"))
  
  size_pw<-dim(ab)[1]
  
  if (size_pw>1){
	print("one")
    pw_mtx<-matrix(NA, size_pw, size_pw)
    rownames(pw_mtx)<-paste0("entrez_", ab$V1)
    colnames(pw_mtx)<-paste0("entrez_", ab$V1)
    
    if(nrow(ab_edge)!=0){
	print("two")
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
    }
	print("three")
      
      file1<-paste0(paste0("pw_mtx_", j), ".csv")
      filename<-paste0(path1, file1)
      
      write.csv(pw_mtx, filename)
  
  }
}
