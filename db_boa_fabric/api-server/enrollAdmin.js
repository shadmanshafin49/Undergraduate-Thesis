'use strict';
/**
 * enrollAdmin.js
 * Enroll the Fabric CA admin user and store the identity in the wallet.
 * Run this ONCE before starting the API server.
 *
 * Usage:  node enrollAdmin.js
 */
const FabricCAServices = require('fabric-ca-client');
const { Wallets }      = require('fabric-network');
const path             = require('path');
const fs               = require('fs');

async function main() {
    const ccpPath = path.resolve(
        __dirname, '..', '..', 'fabric', 'fabric-samples',
        'test-network', 'organizations', 'peerOrganizations',
        'org1.example.com', 'connection-org1.json'
    );

    if (!fs.existsSync(ccpPath)) {
        console.error('ERROR: Connection profile not found at', ccpPath);
        console.error('Make sure Hyperledger Fabric test-network is installed and running.');
        process.exit(1);
    }

    const ccp    = JSON.parse(fs.readFileSync(ccpPath, 'utf8'));
    const caInfo = ccp.certificateAuthorities['ca.org1.example.com'];
    const caTLSCACerts = caInfo.tlsCACerts.pem;
    const ca     = new FabricCAServices(caInfo.url, {
        trustedRoots: caTLSCACerts, verify: false
    }, caInfo.caName);

    const walletPath = path.join(__dirname, 'wallet');
    const wallet     = await Wallets.newFileSystemWallet(walletPath);

    const identity = await wallet.get('admin');
    if (identity) {
        console.log('Admin identity already in wallet.');
        return;
    }

    const enrollment = await ca.enroll({
        enrollmentID: 'admin', enrollmentSecret: 'adminpw'
    });

    const x509Identity = {
        credentials: {
            certificate: enrollment.certificate,
            privateKey : enrollment.key.toBytes(),
        },
        mspId : 'Org1MSP',
        type  : 'X.509',
    };

    await wallet.put('admin', x509Identity);
    console.log('✔  Admin enrolled and stored in wallet.');
}

main().catch(err => {
    console.error('Enroll admin failed:', err);
    process.exit(1);
});
