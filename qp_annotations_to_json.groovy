import qupath.lib.io.PathIO

def outputPath = "path/to/dataset/annotations/"

def directory = buildFilePath(outputPath)
def folder = new File(directory)
def files =[]

folder.eachFileRecurse() {
    file -> {
        if (file.isFile() && file.getName().contains("qpdata")) {
            files.add(file)
            def hierarchy = PathIO.readHierarchy(file)
            def annotations = hierarchy.getAnnotationObjects()
            def geojsonFilename = file.getName().toString().split('.qpdata')[0]+'.geojson'
            def exportFullPath = new File(file.getAbsoluteFile().getParentFile(), geojsonFilename)
            print annotations
            print "exporting to ${exportFullPath}"
            if (exportFullPath.exists()) {
                exportFullPath.delete()
            }
            exportObjectsToGeoJson(annotations, exportFullPath.toString(), "FEATURE_COLLECTION")
            print "completed ${geojsonFilename}"
        }
    }
}

print "All completed!"
