
if (!window.dash_clientside) {
    window.dash_clientside = {};
}

const SAVE_FILE_NAME = "download_test";


window.dash_clientside.download = {
    rawDownload: function(
        trigger,
        storeData
    ) {
        if (typeof trigger == 'undefined') {
            console.log("Trigger is undefined");
            return false;
        } else if (typeof storeData == 'undefined') {
            console.log("storeData is undefined");
            return false;
        }
        console.log("We're parsin'!");
        // generate file and send through file-saver
        storeJson = JSON.parse(storeData);
        //console.log("The JSON looks like this");
        //console.log(storeJson);
        // Get the column names
        var columnNames = storeJson.columns;//Object.keys(storeJson);
        //let firstCol = columnNames[0];
        // Get the indices of the df
        var idx = storeJson.index;//Object.keys(storeJson[firstCol]);
        //console.log('these are the indexes');
        //console.log(idx);
        // get the data!
        // define a function that return an empty cell instead of a null
        var replacer = function(key, value) {
            return value === null ? '' : value;
        };
        // csv variable is a long string. Lines are cut by \n character
        var firstLine = [];
        firstLine.push('index');
        firstLine = firstLine.concat(columnNames);
        firstLine.push('\n');
        firstLine = firstLine.join(',');
        //console.log('this is the first line');
        //console.log(firstLine);
        var csv = firstLine;
        //function getPoint(item, line_idx) {
        //    return storeJson[item][line_idx];
        //}
        for (var i = 0; i < idx.length; i++) {
            let line=[];
            let line_idx = idx[i];
            line.push(line_idx);
            line.push(storeJson.data[i]);
            line.push('\n');
            line = line.join(',');
            csv = csv.concat(line);
            // more statements
           }
        //console.log(csv);
        const file = new Blob([csv], {
            type: "text/csv;charset=utf-8"
        });

        //console.log("downloading figure data to csv.");
        saveAs(file, SAVE_FILE_NAME + ".csv");
    },

    multiDownload: function(
        trigger,
        storeData
    ) {
        return this.rawDownload(
            trigger,
            storeData
        );
    },

    uniDownload: function(
        trigger,
        sensorStore
    ) {
        if (typeof trigger == 'undefined') {
            console.log("Trigger is undefined");
            return false;
        } else if (typeof sensorStore == 'undefined') {
            console.log("storeData is undefined");
            return false;
        }
        // generate file and send through file-saver
        storeJson = JSON.parse(sensorStore);
        
        console.log("The JSON looks like this");
        console.log(storeJson);
    }

};