'use strict';

const DBBOAContract = require('./lib/db_boa_chaincode');

module.exports.DBBOAContract = DBBOAContract;
module.exports.contracts     = [DBBOAContract];
