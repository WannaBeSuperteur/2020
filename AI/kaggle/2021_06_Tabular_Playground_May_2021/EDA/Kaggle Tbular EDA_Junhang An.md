+ **Kaggle Tbular EDA**

  **The content from the [Kaggle:Herose](https://www.kaggle.com/piantic/tps-june-2021-basic-eda-t-sne-visualization)** **and also change the data with May-2021**

  ![image-20210621123021188](C:\Users\AnJunkang\AppData\Roaming\Typora\typora-user-images\image-20210621123021188.png)

  The above picture is about the *sample_submission.csv* it will let us predict the each **class**'s probablity for every **id**.And it is the target of this competition.

  + **General EDA**

    ![image-20210621124755373](C:\Users\AnJunkang\AppData\Roaming\Typora\typora-user-images\image-20210621124755373.png)

    + **Basic EDA**

      + **Target Distribution**

        ![](C:\Users\AnJunkang\AppData\Roaming\Typora\typora-user-images\image-20210621130700253.png)

        ```
        Class_1     8490
        Class_2    57497
        Class_3    21420
        Class_4    12593
        ```

        From this distribution graph I think the data have too many bias. And it will be a big problem of the training.

        Also check the range from 'feature_0' to 'feature_49'

        <img src="C:\Users\AnJunkang\AppData\Roaming\Typora\typora-user-images\image-20210621132801639.png" alt="image-20210621132801639"  />

        ![image-20210621132942541](C:\Users\AnJunkang\AppData\Roaming\Typora\typora-user-images\image-20210621132942541.png)

      **Make a statistics in train data and test data**

      First, let make the difference between two dataset.

      ![image-20210621133309305](C:\Users\AnJunkang\AppData\Roaming\Typora\typora-user-images\image-20210621133309305.png)

      The above picture shows the difference between train data and test data.

      The follow picutre shows the Features_0 to Feature_49 range, the green block means Trian>Test, the red block means Train<Test.

      ![image-20210621133808615](C:\Users\AnJunkang\AppData\Roaming\Typora\typora-user-images\image-20210621133808615.png)

      The following picture shows the sorted Feature range.

      ![image-20210621133903041](C:\Users\AnJunkang\AppData\Roaming\Typora\typora-user-images\image-20210621133903041.png)

       Also check the kdeplot, the following picture shows:

      ![image-20210621134625164](C:\Users\AnJunkang\AppData\Roaming\Typora\typora-user-images\image-20210621134625164.png)

      The zeor value by feature:

      ![image-20210621134747565](C:\Users\AnJunkang\AppData\Roaming\Typora\typora-user-images\image-20210621134747565.png)

      The average by feature(Blue=Class_1,Yellow=Class_2,Green=Class_3,Red = Class_4)

      ![image-20210621134900796](C:\Users\AnJunkang\AppData\Roaming\Typora\typora-user-images\image-20210621134900796.png)

      The correlation between Feature_0 to Feature_49

      ![image-20210621134941946](C:\Users\AnJunkang\AppData\Roaming\Typora\typora-user-images\image-20210621134941946.png)

      

