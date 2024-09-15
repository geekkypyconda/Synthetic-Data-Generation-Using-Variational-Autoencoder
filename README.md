# Synthetic-Data-Generation-Using-Variational-Autoencoder
Developed a Variational Autoencoder (VAE) model to generate 2 million synthetic card transaction records, focusing on data preprocessing, model design, and evaluation using KL Divergence. Ensured data quality through scaling and loss function optimization, achieving realistic data distribution comparisons.

## Steps Used for generating Synthetic Data

### Importing Data:
We used pandas for reading the card transaction csv file

### Pre-processing data:

1. We removed the dollar sign from the amount column to make it into a numeric value
2. We then used Label Encoder to encode all the columns except Amount col.
3. Then we fill all the NaN values with the mean of the data using the data.fillna(data.mean()).
4. Then we use Min Max scaler to scale all the values and for that
5. We first select all the columns which have numeric value(either float or int) 
6. Then we do scaler.fit on those data columns
7. Then for each data columns we do scaler.transform
8. Then we will check whether there is a null or infinite value or not in the data set and if there is then we raise a value error.

### Defining the Variational Autoencoder

#### 1. Encoder:
1.a We first add a layer which is of original dimension(15) with activation as relu
1.b Then we add another layer with 12 neurons and activation as relu
1.c Then we add another layer with 10 neurons and activation as relu
1.d Then we add final layer whose dimension is 2 * latent dimension

#### 2. Decoder:
2.a We first add a layer  whose dimension is 2 * latent dimension
2.b Then we add another layer with 10 neurons and activation as relu
2.c Then we add another layer with 12 neurons and activation as relu
2.d Then we add final layer of original dimension with activation as sigmoid

#### 3. Then we made a encoder function(encodeData()) who uses the encoder
3.a First we get the mean log variance using the encoder method
3.b Then we separate mean and log variance
3.c Then we calculate an epsilon(of size latent dimension)
3.d Then we calculate the standard deviation and a value z:
3.e z = mean + eps * standard_deviation
3.f Then we pass this z into the decoder method

#### 4. Reparameterization:
4.a We again calculate an epsilon in the same way as above
4.b Then we find standard deviation : tf.exp(0.5* log_variance)
4.c Then we calculate z = mean + eps * standard_deviation
4.d Then we will return this value z

#### 5. Call method:
We will give input to the encoder and get a result and return the result.

#### 6. Loss calculation function:
6.a We take the input as 
6.a.1 Original Data
6.a.2 Reconstructed data
6.a.3 Mean
6.a.4 Log variance

6.b Then we calculate the reconstruction loss as binary cross entropy.
6.c Then we calculate the KL divergence as
KL_divergence = -0.5 * tf.reduce_mean(1+log_variance-tf.square(mean)-tf.exp(log_variance)

6.d Then we return reconstruction_loss + KL_divergence

#### 7. Compiling & training the model
We put the latent dimension = 4 & using Adam optimizer
Then we used the .fit method for training the model.

#### 8. Generating the Synthetic Data
We generated 2 Million samples.


#### 9. Plotting the distributions : We plotted our distribution as real data vs synthetic data.

#### 10. Calculating the metrics:
We calculated the KL Divergence
