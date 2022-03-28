import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class RealEstateModel(pl.LightningModule):
    def __init__(self,
                 encoder_layers_num: int = 3,
                 encoder_outputs_num: int = 7,
                 price_outputs_num: int = 7,
                 offers_outputs_num: int = 7,
                 offers_classes_num: int = 0, # number of classes for outputs 0 if we want to have numeric output, >0 for categorical
                 price_loss_weight: float = 0.5,
                 offers_loss_weight: float = 0.5,
                 dropout_rate=0.2):
        super().__init__()

        self.offers_as_category = offers_classes_num > 0

        # we want to weight losses to indicate the model what is important
        self.price_loss_weight = price_loss_weight
        self.offers_loss_weight = offers_loss_weight

        self.type_linear = nn.LazyLinear(out_features=1, bias=False)
        self.city_linear = nn.LazyLinear(out_features=1, bias=False)
        self.heating_rate_linear = nn.LazyLinear(out_features=1, bias=False)

        self.encoder = nn.ModuleList()
        if encoder_layers_num > 0:
            self.encoder.append(nn.LazyLinear(out_features=encoder_outputs_num))
            # here and below we use ELU function to introduce non-linearity, with reduced vanishing gradient issue
            self.encoder.append(nn.ELU(0.1))
            self.encoder.append(nn.Dropout(dropout_rate))
        for i in range(1, encoder_layers_num):
            self.encoder.append(nn.Linear(in_features=encoder_outputs_num,
                                          out_features=encoder_outputs_num))
            self.encoder.append(nn.ELU(0.1))
            self.encoder.append(nn.Dropout(dropout_rate))

        self.price_head = nn.Sequential(
            nn.LazyLinear(out_features=price_outputs_num),
            nn.ELU(0.1),
            nn.Linear(in_features=price_outputs_num, out_features=1)
        )
        # simple mean squared error, first choice for regression problems
        self.price_loss = F.mse_loss

        self.offers_head = nn.ModuleList(
            [
                nn.LazyLinear(out_features=offers_outputs_num),
                nn.Sigmoid()
            ]
        )
        if self.offers_as_category:
            self.offers_head.append(nn.Linear(in_features=offers_outputs_num, out_features=offers_classes_num))
            # we will get as outputs probabilities for classes:
            self.offers_head.append(nn.Softmax())
            # categorical x-entropy to deal with predictions for different classes:
            self.offers_loss = F.cross_entropy
        else:
            self.offers_head.append(nn.Linear(in_features=offers_outputs_num, out_features=1))
            self.offers_loss = F.mse_loss

    def forward(self, x):
        # One can have all numeric inputs already concatenated. Here we keep them separately for readability
        # Here we assume type_, city, heating_rate being already one-hot encoded
        type_, city, heating_rate, transport_distance, size, num_rooms, num_bathrooms = x

        type_value = self.type_linear(type_)
        city_value = self.city_linear(city)
        heating_rate_value = self.heating_rate_linear(heating_rate)

        avg_room_size = size / num_rooms

        xxx = torch.concat(
            [
                type_value,
                city_value,
                heating_rate_value,
                transport_distance,
                avg_room_size,
                num_bathrooms
            ],
            dim=-1
        )

        for layer in self.encoder:
            xxx = layer(xxx)

        price_log = self.price_head(xxx)

        offers = xxx
        for layer in self.offers_head:
            offers = layer(xxx)

        return price_log, offers

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        sale_price, offers = targets

        predicted_sale_price_log, predicted_offers = self(inputs)

        loss_for_price_log = self.price_loss(predicted_sale_price_log, torch.log(sale_price))

        # if offers are categorical should be already one-hot encoded
        loss_for_offers = self.offers_loss(predicted_offers, offers)

        # returning weighted loss
        return self.price_loss_weight * loss_for_price_log + self.offers_loss_weight * loss_for_offers

    def configure_optimizers(self):
        # Using adaptive optimizer for faster convergence
        return torch.optim.Adam(self.parameters(), lr=0.001)


if __name__ == "__main__":
    model = RealEstateModel()
