'use strict';
/**
 * registerUser.js
 * Register and enroll "appUser" under Org1.
 * Run this ONCE after enrollAdmin.js.
 *
 * Usage:  node registerUser.js
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
        console.error('Connection profile not found:', ccpPath);
        process.exit(1);
    }

    const ccp    = JSON.parse(fs.readFileSync(ccpPath, 'utf8'));
    const caInfo = ccp.certificateAuthorities['ca.org1.example.com'];
    const ca     = new FabricCAServices(caInfo.url, {
        trustedRoots: caInfo.tlsCACerts.pem, verify: false
    }, caInfo.caName);

    const walletPath = path.join(__dirname, 'wallet');
    const wallet     = await Wallets.newFileSystemWallet(walletPath);

    const userIdentity = await wallet.get('appUser');
    if (userIdentity) {
        console.log('appUser already registered.');
        return;
    }

    const adminIdentity = await wallet.get('admin');
    if (!adminIdentity) {
        console.error('Admin not found. Run enrollAdmin.js first.');
        process.exit(1);
    }

    const provider = wallet.getProviderRegistry().getProvider(adminIdentity.type);
    const adminUser = await provider.getUserContext(adminIdentity, 'admin');

    const secret = await ca.register({
        affiliation : 'org1.department1',
        enrollmentID: 'appUser',
        role        : 'client',
    }, adminUser);

    const enrollment = await ca.enroll({
        enrollmentID: 'appUser', enrollmentSecret: secret
    });

    const x509Identity = {
        credentials: {
            certificate: enrollment.certificate,
            privateKey : enrollment.key.toBytes(),
        },
        mspId : 'Org1MSP',
        type  : 'X.509',
    };

    await wallet.put('appUser', x509Identity);
    console.log('✔  appUser registered and stored in wallet.');
}

main().catch(err => {
    console.error('Register user failed:', err);
    process.exit(1);
});
