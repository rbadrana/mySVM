package main

import (
	"math"
	"math/rand"

	"github.com/gonum/stat"
	"gonum.org/v1/gonum/mat"
)

type SVM struct {
	degree float64
	C      float64
	alpha  []float64
	b      float64
	X      *mat.Dense
	Y      []float64
	gamma  float64
	coef0  float64
}

func (svm SVM) RBFKernel(x1, x2 []float64) float64 {
	gamma := svm.gamma
	norm := 0.0
	for i := 0; i < len(x1); i++ {
		norm += math.Pow(x1[i]-x2[i], 2)
	}
	return math.Exp(-gamma * norm)
}

func (svm SVM) PolyKernel(x1, x2 []float64) float64 {
	dot := 0.0
	for i := 0; i < len(x1); i++ {
		dot += x1[i] * x2[i]
	}
	value := svm.C + dot
	return math.Pow(value, float64(svm.degree))
}

func (svm SVM) SigmoidKernel(x1, x2 []float64) float64 {
	dot := 0.0
	for i := 0; i < len(x1); i++ {
		dot += x1[i] * x2[i]
	}
	value := svm.gamma*dot + svm.coef0
	return math.Tanh(value)
}

func (svm *SVM) Train(x [][]float64, y []float64) {
	//rand.Seed(time.Now().UnixNano())

	svm.X = mat.NewDense(len(x), len(x[0]), nil)
	for i := 0; i < len(x); i++ {
		for j := 0; j < len(x[0]); j++ {
			svm.X.Set(i, j, x[i][j])
		}
	}

	svm.Y = y

	alpha := make([]float64, len(x))
	for i := 0; i < len(alpha); i++ {
		alpha[i] = 0
	}

	xMat := mat.NewDense(len(x), len(x[0]), nil)
	for i := 0; i < len(x); i++ {
		for j := 0; j < len(x[0]); j++ {
			xMat.Set(i, j, x[i][j])
		}
	}

	gramMatrix := mat.NewDense(len(x), len(x), nil)
	for i := 0; i < len(x); i++ {
		for j := 0; j < len(x); j++ {
			gramMatrix.Set(i, j, svm.PolyKernel(x[i], x[j]))
			//gramMatrix.Set(i, j, svm.RBFKernel(x[i], x[j]))
			//gramMatrix.Set(i, j, svm.sigmoidKernel(x[i], x[j]))
		}
	}

	var supportVectors []int
	for epoch := 0; epoch < 100; epoch++ {
		for i := 0; i < len(x); i++ {
			f := mat.Dot(mat.NewVecDense(len(x), alpha), mat.NewVecDense(len(x), svm.Y))
			if svm.Y[i]*f < 1 {
				alpha[i] += svm.C - svm.Y[i]*f
			}
		}

		if epoch%10 == 0 {
			for i := 0; i < len(x); i++ {
				if alpha[i] > 0 {
					supportVectors = append(supportVectors, i)
				}
			}
		}
	}

	for i := 0; i < len(x); i++ {
		alpha[i] *= svm.Y[i]
	}
	svm.alpha = alpha

	// Compute the value of svm.b
	var bSum float64
	numSupportVectors := float64(len(supportVectors))
	for _, i := range supportVectors {
		bSum += svm.Y[i] - mat.Dot(mat.NewVecDense(len(x), svm.alpha), gramMatrix.RowView(i))
	}
	svm.b = bSum / numSupportVectors
}

func (svm SVM) Predict(testdata []float64) float64 {
	result := 0.0

	for i := 0; i < len(svm.alpha); i++ {
		dotProduct := svm.PolyKernel(svm.X.RawRowView(i), testdata)
		//dotProduct := svm.RBFKernel(svm.X.RawRowView(i), testdata)
		//dotProduct := svm.sigmoidKernel(svm.X.RawRowView(i), testdata)
		result += svm.alpha[i] * svm.Y[i] * dotProduct
	}
	result += svm.b
	//return result
	if result >= 0.5 {
		return 1.0
	} else {
		return 0.0
	}

}

//scaling functions

func ScaleFeatures(x [][]float64) [][]float64 {
	scaled := make([][]float64, len(x))
	for i := range scaled {
		scaled[i] = make([]float64, len(x[i]))
	}
	for j := range x[0] {
		col := make([]float64, len(x))
		for i := range x {
			col[i] = x[i][j]
		}
		mean, std := stat.MeanStdDev(col, nil)
		for i := range x {
			scaled[i][j] = (x[i][j] - mean) / std
		}
	}
	return scaled
}

func ScaleLabels(y []float64) []float64 {
	scaled := make([]float64, len(y))
	mean, std := stat.MeanStdDev(y, nil)
	for i := range y {
		scaled[i] = (y[i] - mean) / std
	}
	return scaled
}

// Spliting Data for checking accuracy
func Split(x [][]float64, y []float64, ratio float64, seed int64) (train_x, test_x [][]float64, train_y, test_y []float64) {
	// Set the seed for the random number generator
	source := rand.NewSource(seed)
	r := rand.New(source)

	// Calculate the number of training examples
	nTrain := int(math.Round(float64(len(x)) * (1.0 - ratio)))

	// Shuffle the indices of the examples
	indices := r.Perm(len(x))

	// Split the indices into training and test indices
	trainIndices := indices[:nTrain]
	testIndices := indices[nTrain:]

	// Initialize the training and test sets
	train_x = make([][]float64, nTrain)
	train_y = make([]float64, nTrain)
	test_x = make([][]float64, len(x)-nTrain)
	test_y = make([]float64, len(x)-nTrain)

	// Fill in the training and test sets
	for i, idx := range trainIndices {
		train_x[i] = x[idx]
		train_y[i] = y[idx]
	}
	for i, idx := range testIndices {
		test_x[i] = x[idx]
		test_y[i] = y[idx]
	}

	return train_x, test_x, train_y, test_y
}

func Accuracy(predicted []float64, actual []float64) float64 {
	n := len(predicted)
	nCorrect := 0
	for i := 0; i < n; i++ {
		if predicted[i] == actual[i] {
			nCorrect++
		}
	}
	return (float64(nCorrect) / float64(n)) * 100
}
