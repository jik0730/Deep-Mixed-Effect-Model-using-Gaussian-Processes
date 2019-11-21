# Deep Mixed Effect Model using Gaussian Processes: A Personalized and Reliable Prediction for Healthcare
+ Jihun Yun (KAIST), Peng Zheng (University of Washington), Eunho Yang (KAIST, AITRICS), Aurélie C. Lozano (IBM T.J.
Watson Research Center), and Aleksandr Aravkin (University of Washington)

This repository contains implementations for our **AAAI 2020** (to appear) paper "Deep Mixed Effect Model using Gaussian Processes: A Personalized and Reliable Prediction for Healthcare".

# Abstract

We present a personalized and reliable prediction model for healthcare, which can provide individually tailored medical services such as diagnosis, disease treatment, and prevention. Our proposed framework targets at making personalized and reliable predictions from time-series data, such as Electronic Health Records (EHR), by modeling two complementary components: i) a shared component that captures global trend across diverse patients and ii) a patient-specific component that models idiosyncratic variability for each patient. To this end, we propose a composite model of a deep  neural network to learn complex global trends from the large number of patients, and Gaussian Processes (GP) to probabilistically model individual time-series given relatively small number of visits per patient. We evaluate our model on diverse and heterogeneous tasks from EHR datasets and show practical advantages over standard time-series deep models such as pure Recurrent Neural Network (RNN).

<!-- # Citation

If you think this repo is helpful, please cite as
```
@inproceedings{yun2019trimming,
  title={Trimming the $$\backslash$ell\_1 $ Regularizer: Statistical Analysis, Optimization, and Applications to Deep Learning},
  author={Yun, Jihun and Zheng, Peng and Yang, Eunho and Lozano, Aurelie and Aravkin, Aleksandr},
  booktitle={International Conference on Machine Learning},
  pages={7242--7251},
  year={2019}
}
``` -->