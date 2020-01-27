plt.figure(figsize=(10,10))
for i in range(10):   
    plt.subplot(1, 10, i+1)
    plt.imshow(dataset.images[i], cmap='gray')
