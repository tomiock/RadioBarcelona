# RadioBarcelona

## KPIs
- processing of all or a very large quantity of the dataset (>50%)
- test set with small quantity of pages (5-10) that can be used for qualitative results
- test set with small quantity of pages (5-10) to check for OCR quality (quantitative metric)
- test set with small quantity of pages (5-10) to check for object recognition metrics (60% mAP[50])

## Requirements
- web interface with decent performance over a large quantity of pages (>10k)
- open source pipeline to replicate or adapt to other types of historical documents

# Project Structure

## WP 1 Dataset Preparation
#### Task 1.1 Qualitative description of dataset
- Idenfity how to the pages from different years differ
#### Task 1.2 Quantitative description of dataset
- Count the total number of pages and the necessary storage

## WP 2 Data Processing
#### Task 2.1 Testing of Suitable Initial Pipeline
- Identication of candidates
- Test end evaluate outputs (vibecheck)
#### Task 2.2 Definition and Development of Pipeline
#### Task 2.3 Preparation of long-running execution
- Specifiying the formats used
- Performance metrics (pages/s)
#### Task 2.4 Data Processing
- Run process over dataset
- Test with developed metrics

## WP 3 Evaluation
#### Task 3.1 Development of Test Sets (Ground Truth)
- Clear definition of target data (e.g. pages with "sellos")
- Manual selection of data based on targets (remove them from database)
- Development of the datasets via manual/AI annotations
#### Task 3.2 Development of Metrics and Evaluation Pipeline
- Development of metrics with mock data
- Development of evaluation pipeline (processes test set and compare with ground truth)

## WP 4 Final Deliverable
#### Task 4.1 Development of Webinterface according to defined formats
- Allow the displaying of all generated data

#### Task 4.2 Stress-Testing of Webinterface
- Test with 10k pages
