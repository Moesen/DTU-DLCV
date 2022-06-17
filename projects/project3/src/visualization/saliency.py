
from keras import backend as K
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.scores import CategoricalScore

logits = new_model(img)
class_pred = predicted[0]

score = CategoricalScore([class_pred])

saliency = Saliency(new_model, clone=True)

saliency_map = saliency(
    score,
    img,
    smooth_samples=20,  # The number of calculating gradients iterations.
    smooth_noise=0.20,
)  # noise spread level.

fig, axs = plt.subplots(1,2,figsize=(15,8))

axs[0].imshow(img.numpy().squeeze())

axs[1].imshow(img.numpy().squeeze())
axs[1].imshow(saliency_map.squeeze(), cmap=cmap, alpha=0.5)
saliency_fig_path = PROJECT_ROOT / "reports/figures/smoothgrad_saliency.png"
plt.savefig(saliency_fig_path)
