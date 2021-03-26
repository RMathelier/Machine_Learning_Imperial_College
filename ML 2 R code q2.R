

########## Machine Learning - Coursework 2 ##########

setwd("C:/Users/robin/Dropbox/Applications/Overleaf/Coursework 2 ML")
width=16
height=10

library(caret)
library(ggplot2)
library(tidyr)
library(matrixStats)
library(purrr)
library(NbClust)
library(graphics)
library(dendextend)
library(GGally)
library(pracma)
library(fossil)
library(gridExtra)
library(amap)
library(cluster)

set.seed(42)

### load data ###

data = read.csv("CID1945214.csv")

head(data)
summary(data)

n = dim(data)[1]
p = dim(data)[2]

head(data)
summary(data)

### NA ###

sum(sapply(1:n, function(i) sum(is.na(data[i,]))))
lapply(1:p, function(j) data[is.na(data[,j]),j] <<- mean(data[,j], na.rm = TRUE))
summary(data)


####### Hierarchical clustering #########

features = data[,-1]
data_t = t(data[,-1])
data_t_scale = scale(data_t)


diss_matrix_spear = 1 - abs(cor(features, method = "pearson"))
diss_matrix_spear = as.dist(diss_matrix_spear)

# plot the resulting dendrograms

pdf("dends.pdf", width = width, height = height/2)

par(mfrow = c(2,3))

dend_sin_cor = hclust(diss_matrix_spear, method="single")
dend_avg_cor = hclust(diss_matrix_spear, method="average")
dend_comp_cor = hclust(diss_matrix_spear, method="complete")

plot(dend_sin_cor,
     main = "Spearman correlation and single linkage",
     cex.main = 1.5,
     xlab="Features")

plot(dend_avg_cor,
     main = "Spearman correlation and average linkage",
     cex.main = 1.5,
     xlab="Features")

plot(dend_comp_cor,
     main = "Spearman correlation and complete linkage",
     cex.main = 1.5,
     xlab="Features")

dend_sin_euc = hclust(dist(data_t_scale, method = "euclidean"), method="single")
dend_avg_euc = hclust(dist(data_t_scale, method = "euclidean"), method="average")
dend_comp_euc = hclust(dist(data_t_scale, method = "euclidean"), method="complete")



plot(dend_sin_euc,
     main = "Euclidean distance and single linkage",
     cex.main = 1.5,
     xlab="Features")

plot(dend_avg_euc,
     main = "Euclidean distance and average linkage",
     cex.main = 1.5,
     xlab="Features")

plot(dend_comp_euc,
     main = "Euclidean distance and complete linkage",
     cex.main = 1.5,
     xlab="Features")

dev.off()


##### Seek appropriate number of clusters #####

nbc_comp <- NbClust(data_t_scale,
                        min.nc=2,
                        max.nc=27, 
                        method="complete",
                        diss = diss_matrix_spear,
                        distance = NULL,
                        index="silhouette")$All.index

nbc_sin <- NbClust(data_t_scale,
                       min.nc=2,
                       max.nc=27,
                       method="single",
                       diss = diss_matrix_spear, 
                       distance = NULL,
                       index="silhouette")$All.index


nbc_avg <- NbClust(data_t_scale, 
                       min.nc=2,
                       max.nc=27, 
                       method="average",
                       diss = diss_matrix_spear,
                       distance = NULL,
                       index="silhouette")$All.index


sil_sin_cor <- sapply(2:27, function(i) {summary(silhouette(cutree(dend_sin_cor, k = i), 
                                                           diss_matrix_spear, full = FALSE))$avg.width})

sil_comp_cor <- sapply(2:27, function(i) {summary(silhouette(cutree(dend_comp_cor, k = i), 
                                                            diss_matrix_spear, full = FALSE))$avg.width})

sil_avg_cor <- sapply(2:27, function(i) {summary(silhouette(cutree(dend_avg_cor, k = i), 
                                                             diss_matrix_spear, full = FALSE))$avg.width})

pdf(file = "silhouette_hier.pdf")

data.frame(n_clusters=2:27, Complete = sil_comp_cor,
           Single = sil_sin_cor, Average = sil_avg_cor) %>%
  pivot_longer(-n_clusters, names_to="method", values_to="index_value") %>%
  ggplot(aes(x=n_clusters, y=index_value, colour=method)) + geom_line() + geom_point() + 
  labs(x = "Number of clusters K", y = "Silhouette measure")

dev.off()

# Average + Absolute correlation

pdf(file = "dendrogram.pdf")

par(mfrow = c(1,1))

dend = hclust(diss_matrix_spear, method="average")

plot(dend,
     main="Spearman Correlation distance and average linkage",
     xlab="Features")
rect.dendrogram(as.dendrogram(dend), 
                k = 5, 
                border = c(2:9, "navyblue", "orange"), 
                lwd = 2, 
                lower_rect = -0.2)
dev.off()

########## k-mean ##########

library("LICORS")

data_scale = scale(data)
data_scale = data_scale[,-1]

pdf("silhouette_k_mean.pdf", width = width/2, height = height/2)
set.seed(42)
nbc_scale <- NbClust(data_scale, min.nc=2, max.nc=15, method="kmeans", index="silhouette")
data.frame(n_clusters=2:15,  nbc_scale$All.index) %>%
  pivot_longer(-n_clusters, names_to="method", values_to="index_value") %>%
  ggplot(aes(x=n_clusters, y=index_value), col="red") + geom_line() + geom_point() + 
  labs(x = "Number of clusters K", y = "Silhouette measure")
dev.off()

set.seed(42)
km_rep = replicate(10000, {km <- kmeans(data_scale, centers = 2, iter.max = 10)
c(km$tot.withinss, dist(km$centers))})

with_rep = km_rep[1,]
dist_centroids = km_rep[2,]

summary(with_rep)
summary(dist_centroids)

best_tot_withinss = Inf
n_iter = 10000
i = 0
for (i in 1:n_iter){
  km = kmeans(data_scale, centers = 2, iter.max = 10)
  if (km$tot.withinss < best_tot_withinss){
    best_dist_centroids = km$tot.withinss
    km_final = km
  }
}

km_final$tot.withinss
dist(km_final$centers)

pca = prcomp(data_scale, retx = TRUE, rank. = 4)

df <- as.data.frame(cbind(pca$x, km_final$cluster))
colnames(df)[[5]] <- "cluster_label"
df$cluster_label <- as.factor(df$cluster_label)

ggpairs(df, columns=1:4, aes(colour=cluster_label), progress=FALSE)


##### Rand index #####

round(rand.index(km_final$cluster, data$z), digits = 4)

df <- as.data.frame(cbind(pca$x, km_final$cluster))
colnames(df)[[5]] <- "cluster_label"
df$cluster_label <- as.factor(- df$cluster_label + 2)
df$z = as.factor(data$z)

pdf("pca_z.pdf", width = width/2, height = height/2)

g1 = ggplot(df, aes(x = PC1, y = PC2, colour=cluster_label), progress=FALSE) +
  geom_point(alpha = 0.4) + theme(legend.position="bottom")
g2 = ggplot(df, aes(x = PC1, y = PC2, colour=z), progress=FALSE) +
  geom_point(alpha = 0.4) + theme(legend.position="bottom")
grid.arrange(g1, g2, ncol = 2)

dev.off()




##### k-mean ++ #####

library("flexclust")

kpp = kcca(data_scale, k = 2, family = kccaFamily("kmeans"), control=list(initcent="kmeanspp"),
           simple = FALSE, save.data = TRUE)

info(kpp, which = "distsum")  # sum withinss
info(kpp, which = "av_dist")
cluster_label_pp = predict(kpp)
predict(kpp)

head(ddata_kpp)

sum(cluster_label_pp != km_final$cluster)

# final km 2D

df_pp <- as.data.frame(cbind(pca$x, cluster_label_pp))
colnames(df_pp)[[5]] <- "cluster_label"
df_pp$cluster_label <- as.factor(df_pp$cluster_label)

ggplot(df_pp, aes(x = PC1, y = PC2, colour=cluster_label_pp), progress=FALSE) +
  geom_point()

