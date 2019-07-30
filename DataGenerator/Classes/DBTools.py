import mysql.connector

class DBConnector:

    host = "10.11.12.60"
    user = "tisquant"
    passwd = "tisquant"
    db = 'tisquant'
    connection = 0
    cursor = 0

    def connect(self):
        self.connection = mysql.connector.connect(host=self.host,user=self.user,passwd=self.passwd)
        self.cursor = self.connection.cursor()
        self.cursor.execute('use ' + self.db)

    def execute(self, query=None):
        self.cursor.execute(query)
        return self.cursor.fetchall()

class TisQuantExtract:

    dbconnector = 0

    def __init__(self):
        self.dbconnector = DBConnector()
        self.dbconnector.connect()

    def getLevel3AnnotatedImagesByDiagnosis_Query(self,diagnosis=None,magnification=None,staining_type=None, staining=None, segmentation_function=None, annotator=None, device=None):
        annotator_string = 'and (';
        for i in range(0,annotator.__len__()):
            annotator_string = annotator_string + '(id_annotatedby = ' + str(annotator[i]) + ')'
            if ((i < annotator.__len__()-1) and (annotator.__len__() != 1)):
                annotator_string = annotator_string + ' or '
            else:
                annotator_string = annotator_string + ')'

        diagnosis_string = '('
        for i in range(0,diagnosis.__len__()):
            diagnosis_string = diagnosis_string + ' (diagnosis = \'' + diagnosis[i] + '\')'
            if ((i < diagnosis.__len__()-1) and (diagnosis.__len__() != 1)):
                diagnosis_string = diagnosis_string + ' or '
            else:
                diagnosis_string = diagnosis_string + ')'

        staining_string = 'and ('
        for i in range(0,staining.__len__()):
            staining_string = staining_string + ' (ab_function = \'' + staining[i] + '\')'
            if ((i < staining.__len__()-1) and (staining.__len__() != 1)):
                staining_string = staining_string + ' or '
            else:
                staining_string = staining_string + ')'

        staining_type_string = 'and ('
        for i in range(0,staining_type.__len__()):
            staining_type_string = staining_type_string + ' (staining_type = \'' + staining_type[i] + '\')'
            if ((i < staining_type.__len__()-1) and (staining_type.__len__() != 1)):
                staining_type_string = staining_type_string + ' or '
            else:
                staining_type_string = staining_type_string + ')'

        magnification_string = ' and ('
        for i in range(0,magnification.__len__()):
            magnification_string = magnification_string + ' (magnification = \'' + magnification[i] + '\')'
            if ((i < magnification.__len__()-1) and (magnification.__len__() != 1)):
                magnification_string = magnification_string + ' or '
            else:
                magnification_string = magnification_string + ')'

        query_string = ''
        if (~(staining_type == 'IHC')):
            return 'select distinct image.dbid,image.filepath from image join image_groundtruth_diagnosis on image_groundtruth_diagnosis.image_id = image.dbid where dbid in (select distinct image_id from image_groundtruth_diagnosis where ' + diagnosis_string + staining_type_string + staining_string + magnification_string + ' and type=\'gt_level3\'' + annotator_string + ' and segmentation_function = \'' + segmentation_function + '\' and device = \'' + device + '\')'
        else:
            return ''

    def getLevel3AnnotationByImageId_Query(self,id,annotator):
        annotator_string = 'and (';
        for i in range(0,annotator.__len__()):
            annotator_string = annotator_string + ' (groundtruth.id_annotatedby = ' + str(annotator[i]) + ')'
            if ((i < annotator.__len__() - 1) and (annotator.__len__() != 1)):
                annotator_string = annotator_string + ' or '
            else:
                annotator_string = annotator_string + ')'
        return 'select groundtruth.filepath as gt_filepath from groundtruth join image_groundtruth on image_groundtruth.groundtruth_id =groundtruth.dbid join image on image_groundtruth.image_id =image.dbid join fov on fov.dbid=image.id_fov join slide on slide.dbid = fov.id_slide join sample on sample.dbid=slide.idsample join patient on patient.dbid = sample.patient_id where image.dbid=' + str(id) + ' and groundtruth.type=\'gt_level3\' ' + annotator_string

    def getLevel3AnnotationByImageIdUsingMaxExperience_Query(self,id,annotator):
        annotator_string = 'and (';
        for i in range(0,annotator.__len__()):
            annotator_string = annotator_string + ' (groundtruth.id_annotatedby = ' + str(annotator[i]) + ')'
            if ((i < annotator.__len__() - 1) and (annotator.__len__() != 1)):
                annotator_string = annotator_string + ' or '
            else:
                annotator_string = annotator_string + ')'
        return 'select groundtruth.filepath as gt_filepath from groundtruth join image_groundtruth on image_groundtruth.groundtruth_id =groundtruth.dbid join image on image_groundtruth.image_id =image.dbid join fov on fov.dbid=image.id_fov join slide on slide.dbid = fov.id_slide join sample on sample.dbid=slide.idsample join patient on patient.dbid = sample.patient_id join person on person.dbid = groundtruth.id_annotatedby where image.dbid=' + str(id) + ' and groundtruth.type=\'gt_level3\' ' + annotator_string + 'and person.yearsofexperience = (select max(yearsofexperience) from groundtruth join image_groundtruth on image_groundtruth.groundtruth_id =groundtruth.dbid join image on image_groundtruth.image_id =image.dbid join fov on fov.dbid=image.id_fov join slide on slide.dbid = fov.id_slide join sample on sample.dbid=slide.idsample join patient on patient.dbid = sample.patient_id join person on person.dbid = groundtruth.id_annotatedby where image.dbid=' + str(id) + ' and groundtruth.type=\'gt_level3\' ' + annotator_string + ')'

    def getLevel2AnnotationByImageIdUsingMaxExperience_Query(self,id,annotator):
        annotator_string = 'and (';
        for i in range(0,annotator.__len__()):
            annotator_string = annotator_string + ' (groundtruth.id_annotatedby = ' + str(annotator[i]) + ')'
            if ((i < annotator.__len__() - 1) and (annotator.__len__() != 1)):
                annotator_string = annotator_string + ' or '
            else:
                annotator_string = annotator_string + ')'
        return 'select groundtruth.filepath as gt_filepath from groundtruth join image_groundtruth on image_groundtruth.groundtruth_id =groundtruth.dbid join image on image_groundtruth.image_id =image.dbid join fov on fov.dbid=image.id_fov join slide on slide.dbid = fov.id_slide join sample on sample.dbid=slide.idsample join patient on patient.dbid = sample.patient_id join person on person.dbid = groundtruth.id_annotatedby where image.dbid=' + str(id) + ' and groundtruth.type=\'gt_level2\' ' + annotator_string + 'and person.yearsofexperience = (select max(yearsofexperience) from groundtruth join image_groundtruth on image_groundtruth.groundtruth_id =groundtruth.dbid join image on image_groundtruth.image_id =image.dbid join fov on fov.dbid=image.id_fov join slide on slide.dbid = fov.id_slide join sample on sample.dbid=slide.idsample join patient on patient.dbid = sample.patient_id join person on person.dbid = groundtruth.id_annotatedby where image.dbid=' + str(id) + ' and groundtruth.type=\'gt_level3\' ' + annotator_string + ')'
