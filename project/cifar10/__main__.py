#!python3.8
import torch
import torchvision
from torchvision import datasets, transforms

import numpy as np
import pandas as pd
# 主成分分析用
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import KernelPCA

from utils.plot_save import scatter_plot

device_type = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"device type = {device_type}")

dtype = torch.float
device = torch.device(device_type)

cifar10_labels = (
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck')

# 一般的な画像変換で、複数の変換を一緒に構成します。（変換作成するリスト。）
transform = transforms.Compose([
    transforms.ToTensor() # <[0, 255]の範囲のnumpy.ndarray(H,W,C)> を <[0.0, 1.0]の範囲のtorch.FloatTensor(C, H, W)>に変換
    ])

cifar_10_dataset_train = datasets.CIFAR10(
    root='./data',
    train=True, #トレーニングセットからデータセットを作成
    download=True,
    transform=transform # PIL画像を取り込んで変換されたバージョンを返す関数/変換
    )

# データセットとサンプラーを組み合わせ、指定されたデータセットの反復可能なオブジェクトを提供する。
cifar_10_dataset_train_loader = torch.utils.data.DataLoader(
    dataset=cifar_10_dataset_train,
    batch_size = 10,
    shuffle=True,
    pin_memory=True, # CUDAのピンメモリを使用可否。
    timeout=0, #ワーカーからバッチを収集する為のタイムアウト値
    drop_last=True, # Trueの場合バッチサイズに割り切れない不完全なバッチを削除する。
    num_workers=2 #0の場合メインスレッドで行う。
)
print(f'cifar_10_dataset_train type {type(cifar_10_dataset_train)}')
print(f'len(cifar_10_dataset_train)={len(cifar_10_dataset_train)}')

# データセット取得(アンパック)n
image_data, image_type = cifar_10_dataset_train[0]
print(f'type {type(image_data)}')

# 特徴量を抽出(キャッシュ)
im_channel, *im_resolution = image_data.shape
print(f'im_channel:{im_channel}, im_resolution:{im_resolution}')

#特徴のサイズを計算
channel_feature_size = torch.prod(torch.tensor(im_resolution)).item() * im_channel
print(f'feature size : {channel_feature_size}')

# データローダーからtupleへ変換
images, categorys = zip(*[ other for other in cifar_10_dataset_train])
# 特徴量を[C, H, W]を一次元化
images = np.array([ np.ravel(img.to('cpu').detach().numpy().copy()) for img in images])
print(f'images shape {images.shape}')

# CIFAR10のデータセットをDataframeに変換
categorys_ds = pd.Series(data=categorys, index=range(0, images.shape[0]))
image_df = pd.DataFrame(data=images, index=range(0, images.shape[0]))
print(f'imgs_df {image_df.shape}')

# _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
# 主成分分析　PCA
# _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
pca = PCA(n_components=channel_feature_size, whiten=False)
cifar_10_dataset_train_pca = pca.fit_transform(image_df)
cifar_10_dataset_train_pca = pd.DataFrame(data=cifar_10_dataset_train_pca, \
                                            index=range(0, cifar_10_dataset_train_pca.shape[0]))

print(f"Variance Explained by all {channel_feature_size} principal components: {sum(pca.explained_variance_ratio_)}")
importanceOfPrincipalComponents = pd.DataFrame(data=pca.explained_variance_ratio_)
importance_data_view = importanceOfPrincipalComponents.T.loc[0:9, :]

scatter_plot(cifar_10_dataset_train_pca, categorys_ds, "PCA")

# _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
# インクリメントPCA
# _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
pca_incremental = IncrementalPCA(n_components=channel_feature_size)
cifar10_dataset_train_incremental_pca = pca_incremental.fit_transform(image_df)
cifar10_dataset_train_incremental_pca = pd.DataFrame(data=cifar10_dataset_train_incremental_pca,\
                                                    index=range(0, cifar10_dataset_train_incremental_pca.shape[0]))

scatter_plot(cifar10_dataset_train_incremental_pca, categorys_ds, "IncrementalPCA")

# _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
# カーネルPCA メモリ不足
# _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
pca_kernel = KernelPCA(n_components=channel_feature_size, kernel='linear')
cifar10_dataset_train_kernel_pca = pca_kernel.fit_transform(image_df.loc[:10000, :])
cifar10_dataset_train_kernel_pca = pd.DataFrame(data=cifar10_dataset_train_kernel_pca, \
                                                index=range(0, cifar10_dataset_train_kernel_pca.shape[0]))

scatter_plot(cifar10_dataset_train_kernel_pca, categorys_ds, "KernelPCA")


# _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
# ISOMAP 全ての点の間の距離をユークリッド距離ではなく曲線距離もしくは測地線距離で計算する
# _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
from sklearn.manifold import Isomap
isomap = Isomap(n_neighbors=5,\
                n_components=channel_feature_size)

isomap.fit(image_df.loc[:10000, :])
cifar10_dataset_train_isomap = isomap.transform(image_df)
cifar10_dataset_train_isomap = pd.DataFrame(data=cifar10_dataset_train_isomap,\
                                            index=range(0, cifar10_dataset_train_isomap.shape[0]))

scatter_plot(cifar10_dataset_train_isomap, categorys_ds, "Isomap")

# _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
# MDS 多次元尺度構成法 データポイント間の類似度を学習し、その類似度を用いて低次元空間にモデル化する。
# _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
from sklearn.manifold import MDS
mds = MDS(n_components=channel_feature_size, metric=True)

cifar10_dataset_train_mds = mds.fit_transform(image_df)
cifar10_dataset_train_mds = pd.DataFrame(data=cifar10_dataset_train_mds, index=range(0, cifar10_dataset_train_mds.shape[0]))

scatter_plot(cifar10_dataset_train_mds, categorys_ds, 'MDS')

# _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
# t-SNE 高次元データを可視化する際に使用される。　類似した点が近くなり、類似していない点が遠くなる。
# 個々の高次元の点を2or3次元空間にモデル化をする。
# _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, learning_rate=300.0)

cifar10_dataset_train_tsne = tsne.fit_transform(image_df.loc[:10000, :])
cifar10_dataset_train_tsne = pd.DataFrame(data=cifar10_dataset_train_tsne, index=cifar10_dataset_train_tsne.index)

scatter_plot(cifar10_dataset_train_tsne, categorys_ds, 'TSNE')