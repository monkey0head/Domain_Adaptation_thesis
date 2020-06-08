# Unsupervised Domain Adaptation thesis
The repo contains a source code for master thesis on Unsupervised Domain Adaptation for Image Processing prepared by Anna Volodkevich, master student of HSE University, Russia, Moscow.  

Исходный код практической части ВКР на тему "Применение методов доменной адаптации для задач обработки изображений", выполненной Володкевич Анной, студенткой образовательной программы
«Науки о данных» факультета компьютерных наук НИУ ВШЭ.

В коде реализованы некоторые методы доменной адаптации для задачи классификации изображений, а именно:
1) Baseline ResNet50 (One Domain Model)
2) DANN: Domain-Adversarial Training of Neural Networks, 2016, https://arxiv.org/abs/1505.07818
3) DANN-CA: Gotta Adapt 'Em All: Joint Pixel and Feature-Level Domain Adaptation for Recognition in the Wild, 2018, https://arxiv.org/abs/1803.00068
4) DADA: Discriminative Adversarial Domain Adaptation, 2019, https://arxiv.org/pdf/1911.12036.

Пример запуска эксперимента обучения модели приведен в файле example.py. В ходе эксперимента модель обучается три раза. Результаты обучения (модель, история изменения метрик и функции потерь и графики для последнего запуска сохраняются локально, а также результаты всех запусков передаются в wandb. Результаты моих экспериментов можно увидеть в wandb: https://app.wandb.ai/monkey_head/domain_adaptation.

Параметры модели устранавливаются в файле configs/dann_config.
В example.py необходимо выбрать архитектуру модели и loss-фукнцию и указать необходимые параметры в configs/dann_config.

Получить эмбеддинги полученной модели можно с помощью get_features.py, оценить качество обученной модели - с помощью evaluate.py.
Необходимо указать путь к файлу обученной модели в аргументе --checkpoint командной строки.

Также в репозитории выложена (презентация)[https://github.com/monkey0head/Domain_Adaptation_thesis/blob/master/presentation.pdf] к дипломной работе, где можно больше узнать о деталях экспериментов.


