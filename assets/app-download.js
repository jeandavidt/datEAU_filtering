
if (!window.dash_clientside) {
    window.dash_clientside = {};
}

const SAVE_FILE_NAME = "download_test";


window.dash_clientside.download = {
    dfToCsv: function(
        dfJson
    ) {
        var columnNames = dfJson.columns;//Object.keys(storeJson);
        //let firstCol = columnNames[0];
        // Get the indices of the df
        var idx = dfJson.index;//Object.keys(storeJson[firstCol]);
        //console.log('these are the indexes');
        //console.log(idx);
        // get the data!
        // define a function that return an empty cell instead of a null
        var replacer = function(value) {
            return value === null ? '' : value;
        };
        // csv variable is a long string. Lines are cut by \n character
        var firstLine = [];
        firstLine.push('datetime');
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
            line.push(dfJson.data[i]);
            line.push('\n');
            line = line.join(',');
            csv = csv.concat(line);
            // more statements
           }
        return csv;
    },

    rawDownload: function(
        trigger,
        storeData
    ) {
        if (typeof trigger == 'undefined') {
            console.log("Raw trigger is undefined");
            return false;
        } else if (typeof storeData == 'undefined') {
            console.log("storeData is undefined");
            return false;
        }
        console.log("We're parsing!");
        // generate file and send through file-saver
        storeJson = JSON.parse(storeData);
        //console.log("The JSON looks like this");
        //console.log(storeJson);
        // Get the column names
        rawCsv = this.dfToCsv(storeJson);
        //console.log(csv);
        const file = new Blob([rawCsv], {
            type: "text/csv;charset=utf-8"
        });

        //console.log("downloading figure data to csv.");
        saveAs(file, SAVE_FILE_NAME + ".csv");
    },

    multiDownload: function(
        trigger,
        storeData
    ) {
        console.log("Function triggered");
        return this.rawDownload(
            trigger,
            storeData
        );
    },

    uniDownload: function(
        trigger,
        channelInfo,
        method,
        sensorStore
    ) {
        if (typeof trigger == 'undefined') {
            console.log("Univariate trigger is undefined");
            return false;
        } else if (typeof sensorStore == 'undefined') {
            console.log("sensorStore is undefined");
            return false;
        }
        //get info to find the channel in the store object
        let splitChannelInfo = channelInfo.split("-");
        console.log('split channel info is');
        console.log(splitChannelInfo);
        let parameterName = splitChannelInfo[3];
        console.log('parameter name is');
        console.log(parameterName);

        // generate file and send through file-saver
        let storeJson = JSON.parse(sensorStore);
        console.log('store Json is');
        console.log(storeJson);
        // extract the correct channel
        let channel = storeJson[0].__Sensor__.channels[parameterName];
        console.log(channel);
        if (channel.__Channel__.filtered == undefined) {
            return false;
        } else {
            let filteredJson = JSON.parse(channel.__Channel__.filtered[method].__DataFrame__);
            console.log("The filtered JSON looks like this");
            console.log(filteredJson);
            let uniCsv = this.dfToCsv(filteredJson);
            //console.log(csv);
            const file = new Blob([uniCsv], {
                type: "text/csv;charset=utf-8"
            });
            //console.log("downloading figure data to csv.");
            saveAs(file, 'univariate_download' + ".csv");
        }
        
    },

};

