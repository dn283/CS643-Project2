package com.aws;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.StructType;

import java.nio.file.Files;
import java.nio.file.Paths;

public class MainWineEngine {

    public static void main(String[] args) throws Exception {

        SparkSession spark = SparkSession.builder()
                .master("local[*]")
                .appName("Predict Wine Quality")
                .getOrCreate();
        JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());
        jsc.setLogLevel("ERROR");

        String file = "D:\\DHRUV_STUDY\\NJIT\\CS 643 CLOUD COMPUTING\\DATASET_PROJECT2\\Schema.json";
        String json = readFileAsString(file);

        StructType schemaFile = (StructType) StructType.fromJson(json);

        Dataset<Row> TrainingDf = spark.read()
                .format("csv")
                .schema(schemaFile)
                .option("header", true)
                .option("delimiter", ";")
                .option("mode", "PERMISSIVE")
                .option("path", "D:\\DHRUV_STUDY\\NJIT\\CS 643 CLOUD COMPUTING\\DATASET_PROJECT2\\TrainingDataset50.csv")
                .load();

        String[] featureCols = new String[]{"fixedAcidity", "volatileAcidity", "citricAcid", "residualSugar", "chlorides", "freeSulfurDioxide",
                "totalSulfurDioxide", "density", "pH", "sulphates", "alcohol"};

        VectorAssembler vectorAssembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features");
        Dataset<Row> vectorDf = vectorAssembler.transform(TrainingDf);
//        System.out.println("Printing vectorDf...");
//        vectorDf.show(10, false);

        StringIndexer indexer = new StringIndexer().setInputCol("quality").setOutputCol("label");
        Dataset<Row> filTrainingDf = indexer.fit(vectorDf).transform(vectorDf);
//        System.out.println("Printing filTrainingDf...");
//        filTrainingDf.show(10, false);

        Dataset<Row> validationDf = spark.read()
                .format("csv")
                .schema(schemaFile)
                .option("header", true)
                .option("escape", "\"")
                .option("delimiter", ";")
                .option("mode", "PERMISSIVE")
                .option("path", "D:\\DHRUV_STUDY\\NJIT\\CS 643 CLOUD COMPUTING\\DATASET_PROJECT2\\ValidationDataset50.csv")
                .load();

//        System.out.println("Printing validationDf...");
//        validationDf.show(10, false);

        Dataset<Row> tranfDf = vectorAssembler.transform(validationDf);
//        System.out.println("Printing tranfDf...");
//        tranfDf.show(10, false);

        Dataset<Row> filValidationDf = indexer.fit(tranfDf).transform(tranfDf);
//        System.out.println("Printing filValidationDf...");
//        filValidationDf.show(10, false);

        DecisionTreeClassifier decisionTreeClassifier = new DecisionTreeClassifier().setMaxDepth(3).setSeed(6030);
        DecisionTreeClassificationModel model = decisionTreeClassifier.fit(filTrainingDf);
        MulticlassClassificationEvaluator eval = new MulticlassClassificationEvaluator().setLabelCol("label");
        Dataset<Row> predictions = model.transform(filValidationDf);
        System.out.println("Printing predictions...");
        predictions.show(10, false);

        System.out.println("Printing predictions label...");
        predictions.select("prediction", "label").show(false);

        double Accuracy = eval.evaluate(predictions);
        System.out.println("Accuracy before: " + Accuracy);

       /* Dataset<Row> wrong = predictions.select("prediction", "label").where("prediction != label");

        System.out.println("Printing wrong count: " + wrong.count());

        long accuracyManual = 1 - (wrong.count() / filValidationDf.count());
        System.out.println("Printing accuracyManual: " + accuracyManual);*/

        MulticlassMetrics multiclassMetrics = new MulticlassMetrics(predictions.select("prediction", "label"));
        System.out.println("Weighted F1 score before pipeline fitting:" + multiclassMetrics.weightedFMeasure());

        ParamMap[] grid = new ParamGridBuilder()
                .addGrid(decisionTreeClassifier.maxBins(), new int[]{15, 25})
                .addGrid(decisionTreeClassifier.maxDepth(), new int[]{5, 10})
                .build();

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{decisionTreeClassifier});

        CrossValidator crossValidator = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(eval)
                .setEstimatorParamMaps(grid)
                .setNumFolds(10);

        CrossValidatorModel crossValidatorModel = crossValidator.fit(filTrainingDf);
        Dataset<Row> predict = crossValidatorModel.transform(filValidationDf);

        double Accuracy2 = eval.evaluate(predict);
        System.out.println("Accuracy after pipeline: " + Accuracy2);

        MulticlassMetrics multiclassMetrics1 = new MulticlassMetrics(predict.select("prediction", "label"));
        System.out.println("Weighted F1 score after:" + multiclassMetrics1.weightedFMeasure());
        System.out.println("Weighted precision:" + multiclassMetrics1.weightedPrecision());
        System.out.println("Weighted recall:" + multiclassMetrics1.weightedRecall());
        System.out.println("Weighted false positive rate:" + multiclassMetrics1.weightedFalsePositiveRate());

    }

    public static String readFileAsString(String file) throws Exception {
        return new String(Files.readAllBytes(Paths.get(file)));
    }
}
